#!/usr/bin/env python
"""Train an Event-JEPA model on packed MEDS events."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from med_jepa_common import (
    cleanup_distributed,
    dist_rank,
    dist_world_size,
    rank0_print,
    set_seed,
    setup_distributed,
    unwrap_model,
    write_json,
)


class PackedEventDataset(Dataset):
    def __init__(self, parquet_path: str | Path) -> None:
        self.table = pq.read_table(parquet_path)

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.table.slice(idx, 1).to_pydict()
        return {
            "input_ids": torch.tensor(row["input_ids"][0], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"][0], dtype=torch.long),
            "event_ids": torch.tensor(row["event_ids"][0], dtype=torch.long),
        }


class Predictor(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4) -> None:
        super().__init__()
        inner = hidden_size * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gather_jepa_pairs(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    event_ids: torch.Tensor,
    eot_token_id: int,
    future_k: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    batch = input_ids.size(0)
    for b in range(batch):
        valid = attention_mask[b].bool()
        if not valid.any():
            continue
        max_event = int(event_ids[b][valid].max().item())
        for event_idx in range(max_event + 1):
            eot_positions = torch.nonzero(valid & (event_ids[b] == event_idx) & (input_ids[b] == eot_token_id), as_tuple=False).flatten()
            if eot_positions.numel() == 0:
                continue
            target_event_hi = min(max_event, event_idx + future_k)
            if target_event_hi <= event_idx:
                continue
            future_mask = valid & (event_ids[b] > event_idx) & (event_ids[b] <= target_event_hi) & (input_ids[b] != eot_token_id)
            if not future_mask.any():
                continue
            sources.append(student_hidden[b, eot_positions[-1]])
            targets.append(teacher_hidden[b, future_mask].mean(dim=0))
    if not sources:
        return None, None
    return torch.stack(sources), torch.stack(targets)


def jepa_loss_fn(pred: torch.Tensor, target: torch.Tensor, var_gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = 2.0 - 2.0 * (F.normalize(pred, dim=-1) * F.normalize(target.detach(), dim=-1)).sum(dim=-1).mean()
    if pred.size(0) < 2 or var_gamma <= 0:
        var_loss = pred.new_zeros(())
    else:
        std = torch.sqrt(pred.float().var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(var_gamma - std).mean().to(pred.dtype)
    return cosine, var_loss


def ar_loss_fn(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, eot_token_id: int, token_weight: float, eot_weight: float) -> torch.Tensor:
    if token_weight <= 0 and eot_weight <= 0:
        return logits.sum() * 0.0
    shift_logits = logits[:, :-1].contiguous()
    targets = input_ids[:, 1:].contiguous()
    valid = attention_mask[:, 1:].bool()
    weights = torch.full_like(targets, token_weight, dtype=shift_logits.dtype)
    weights = torch.where(targets == eot_token_id, torch.full_like(weights, eot_weight), weights)
    weights = weights * valid.to(weights.dtype)
    flat_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1), reduction="none")
    flat_weights = weights.view(-1)
    denom = flat_weights.sum().clamp_min(1.0)
    return (flat_loss * flat_weights).sum() / denom


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    student_model = unwrap_model(student)
    teacher_model = unwrap_model(teacher)
    for s_param, t_param in zip(student_model.parameters(), teacher_model.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


def save_checkpoint(args: argparse.Namespace, model: nn.Module, predictor: nn.Module, tokenizer: Any, name: str) -> None:
    if dist_rank() != 0:
        return
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(unwrap_model(predictor).state_dict(), out / "predictor.pt")
    write_json(out / "training_args.json", vars(args))


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist_world_size() == 1:
        return value.detach()
    x = value.detach().clone()
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x / dist_world_size()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_parquet", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--future_k", type=int, default=1)
    parser.add_argument("--ema_momentum", type=float, default=0.996)
    parser.add_argument("--var_weight", type=float, default=0.0)
    parser.add_argument("--var_gamma", type=float, default=0.02)
    parser.add_argument("--ar_weight", type=float, default=0.03)
    parser.add_argument("--ar_eot_weight", type=float, default=0.0)
    parser.add_argument("--predictor_expansion", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"], default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_distributed()
    set_seed(args.seed + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_token_id = tokenizer.eos_token_id

    student = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    predictor = Predictor(student.config.hidden_size, args.predictor_expansion).to(device=device, dtype=dtype)

    if args.compile:
        student = torch.compile(student)
        predictor = torch.compile(predictor)

    if world > 1:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        predictor = DDP(predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    dataset = PackedEventDataset(args.train_parquet)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    accum_steps = max(1, args.global_batch_size // (args.batch_size * world))
    total_steps = math.ceil(len(loader) * args.epochs / accum_steps)
    optim = torch.optim.AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)

    rank0_print(f"rows={len(dataset)} world={world} accum_steps={accum_steps} total_steps={total_steps}")
    global_step = 0
    student.train()
    predictor.train()
    optim.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            event_ids = batch["event_ids"].to(device, non_blocking=True)

            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            source, target = gather_jepa_pairs(
                student_out.hidden_states[-1],
                teacher_out.hidden_states[-1],
                input_ids,
                attention_mask,
                event_ids,
                eot_token_id,
                args.future_k,
            )
            if source is None or target is None:
                jepa_loss = student_out.logits.sum() * 0.0
                var_loss = student_out.logits.sum() * 0.0
            else:
                pred = predictor(source)
                jepa_loss, var_loss = jepa_loss_fn(pred, target, args.var_gamma)
            ar_loss = ar_loss_fn(student_out.logits, input_ids, attention_mask, eot_token_id, args.ar_weight, args.ar_eot_weight)
            loss = (jepa_loss + args.var_weight * var_loss + ar_loss) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(predictor.parameters()), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                update_ema(student, teacher, args.ema_momentum)
                global_step += 1
                if global_step % args.log_steps == 0:
                    log_loss = reduce_mean(loss.detach() * accum_steps)
                    log_jepa = reduce_mean(jepa_loss.detach())
                    log_ar = reduce_mean(ar_loss.detach())
                    rank0_print(
                        f"epoch={epoch + 1} step={global_step} loss={log_loss.item():.4f} "
                        f"jepa={log_jepa.item():.4f} ar={log_ar.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )
        rank0_print(f"epoch={epoch + 1} done seconds={time.time() - t0:.1f}")
        if args.save_every_epoch:
            save_checkpoint(args, student, predictor, tokenizer, f"epoch_{epoch + 1}")

    save_checkpoint(args, student, predictor, tokenizer, "final")
    cleanup_distributed()


if __name__ == "__main__":
    main()
