#!/usr/bin/env python
"""Train chunk-level JEPA on no-EOT packed EHR token sequences."""

from __future__ import annotations

import argparse
import bisect
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


class PackedTokenDataset(Dataset):
    def __init__(self, parquet_path: str | Path) -> None:
        self.parquet_path = Path(parquet_path)
        self.parquet_file = pq.ParquetFile(self.parquet_path)
        self.row_group_starts: list[int] = []
        total = 0
        for idx in range(self.parquet_file.num_row_groups):
            self.row_group_starts.append(total)
            total += self.parquet_file.metadata.row_group(idx).num_rows
        self.num_rows = total
        self._cached_row_group_idx: int | None = None
        self._cached_table = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["parquet_file"] = None
        state["_cached_row_group_idx"] = None
        state["_cached_table"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.parquet_file = pq.ParquetFile(self.parquet_path)

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row_group_idx = bisect.bisect_right(self.row_group_starts, idx) - 1
        row_group_idx = max(0, row_group_idx)
        row_idx = idx - self.row_group_starts[row_group_idx]
        if self._cached_row_group_idx != row_group_idx:
            self._cached_table = self.parquet_file.read_row_group(
                row_group_idx,
                columns=["input_ids", "attention_mask"],
            )
            self._cached_row_group_idx = row_group_idx
        assert self._cached_table is not None
        row = self._cached_table.slice(row_idx, 1).to_pydict()
        return {
            "input_ids": torch.tensor(row["input_ids"][0], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"][0], dtype=torch.long),
        }


def build_chunk_views(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand each sample into one AR view plus K-1 chunk JEPA views."""
    if num_chunks < 2:
        raise ValueError("--num_chunks must be at least 2")
    batch, seq_len = input_ids.shape
    expanded_ids = input_ids.repeat_interleave(num_chunks, dim=0)
    expanded_mask = attention_mask.repeat_interleave(num_chunks, dim=0).clone()
    source_rows: list[int] = []
    source_pos: list[int] = []
    target_rows: list[int] = []
    target_pos: list[int] = []
    device = input_ids.device
    lengths = attention_mask.sum(dim=1).tolist()

    for b, length_value in enumerate(lengths):
        length = int(length_value)
        base = b * num_chunks
        if length < num_chunks:
            expanded_mask[base + 1 : base + num_chunks].zero_()
            continue
        for chunk_idx in range(1, num_chunks):
            row = base + chunk_idx
            source_end = (chunk_idx * length) // num_chunks - 1
            target_end_exclusive = ((chunk_idx + 1) * length) // num_chunks
            target_end = target_end_exclusive - 1
            if source_end < 0 or target_end <= source_end:
                expanded_mask[row].zero_()
                continue
            if target_end + 1 < seq_len:
                expanded_mask[row, target_end + 1 :].zero_()
            source_rows.append(row)
            source_pos.append(source_end)
            target_rows.append(row)
            target_pos.append(target_end)

    index_dtype = torch.long
    return (
        expanded_ids,
        expanded_mask,
        torch.tensor(source_rows, dtype=index_dtype, device=device),
        torch.tensor(source_pos, dtype=index_dtype, device=device),
        torch.tensor(target_rows, dtype=index_dtype, device=device),
        torch.tensor(target_pos, dtype=index_dtype, device=device),
    )


def ar_loss_fn(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    targets = input_ids[:, 1:].contiguous()
    valid = attention_mask[:, 1:].bool()
    flat_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1), reduction="none")
    flat_valid = valid.view(-1).to(flat_loss.dtype)
    return (flat_loss * flat_valid).sum() / flat_valid.sum().clamp_min(1.0)


def chunk_jepa_loss_fn(
    hidden_states: torch.Tensor,
    source_rows: torch.Tensor,
    source_pos: torch.Tensor,
    target_rows: torch.Tensor,
    target_pos: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if source_rows.numel() == 0:
        zero = hidden_states.sum() * 0.0
        metrics = {
            "cosine_sim": zero.detach(),
            "source_norm": zero.detach(),
            "target_norm": zero.detach(),
            "source_std": zero.detach(),
            "target_std": zero.detach(),
            "pairs": torch.zeros((), dtype=torch.float32, device=hidden_states.device),
        }
        return zero, metrics
    source = hidden_states[source_rows, source_pos]
    target = hidden_states[target_rows, target_pos]
    cosine_sim_per_pair = F.cosine_similarity(source.float(), target.float(), dim=-1)
    loss = 1.0 - cosine_sim_per_pair.mean()
    metrics = {
        "cosine_sim": cosine_sim_per_pair.detach().mean(),
        "source_norm": source.detach().float().norm(dim=-1).mean(),
        "target_norm": target.detach().float().norm(dim=-1).mean(),
        "source_std": source.detach().float().std(dim=0, unbiased=False).mean(),
        "target_std": target.detach().float().std(dim=0, unbiased=False).mean(),
        "pairs": torch.tensor(float(source_rows.numel()), dtype=torch.float32, device=hidden_states.device),
    }
    return loss.to(hidden_states.dtype), metrics


def save_checkpoint(args: argparse.Namespace, model: nn.Module, tokenizer: Any, name: str) -> None:
    if dist_rank() != 0:
        return
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(out)
    tokenizer.save_pretrained(out)
    write_json(out / "training_args.json", vars(args))


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist_world_size() == 1:
        return value.detach()
    x = value.detach().clone()
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x / dist_world_size()


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist_world_size() == 1:
        return value.detach()
    x = value.detach().clone()
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--train_parquet", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--jepa_lambda", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"], default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0, help="Maximum optimizer updates/global batches to train. 0 means full epochs.")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_distributed()
    set_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)

    if args.compile:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    dataset = PackedTokenDataset(args.train_parquet)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader_kwargs: dict[str, Any] = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        drop_last=True,
        **loader_kwargs,
    )
    effective_batch = args.batch_size * world
    accum_steps = max(1, math.ceil(args.global_batch_size / effective_batch))
    updates_per_epoch = len(loader) // accum_steps
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if total_steps <= 0:
        raise ValueError(
            f"No optimizer updates would run: len(loader)={len(loader)} accum_steps={accum_steps} epochs={args.epochs}"
        )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)

    rank0_print(
        f"rows={len(dataset)} world={world} batch_size={args.batch_size} effective_batch={effective_batch} "
        f"accum_steps={accum_steps} updates_per_epoch={updates_per_epoch} total_steps={total_steps} "
        f"max_steps={args.max_steps or 'none'} num_chunks={args.num_chunks} jepa_lambda={args.jepa_lambda}"
    )

    global_step = 0
    update_t0: float | None = None
    total_update_seconds = 0.0
    model.train()
    optim.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        for step, batch in enumerate(loader):
            if global_step >= total_steps or step >= updates_per_epoch * accum_steps:
                break
            if step % accum_steps == 0:
                sync_if_cuda(device)
                update_t0 = time.time()

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            expanded_ids, expanded_mask, source_rows, source_pos, target_rows, target_pos = build_chunk_views(
                input_ids,
                attention_mask,
                args.num_chunks,
            )
            outputs = model(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            ar_logits = outputs.logits[0 :: args.num_chunks]
            ar_loss = ar_loss_fn(ar_logits, input_ids, attention_mask)
            jepa_loss, jepa_metrics = chunk_jepa_loss_fn(
                outputs.hidden_states[-1],
                source_rows,
                source_pos,
                target_rows,
                target_pos,
            )
            loss = (ar_loss + args.jepa_lambda * jepa_loss) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                sync_if_cuda(device)
                global_step += 1
                step_seconds = 0.0 if update_t0 is None else time.time() - update_t0
                total_update_seconds += step_seconds
                if global_step % args.log_steps == 0:
                    log_loss = reduce_mean(loss.detach() * accum_steps)
                    log_ar = reduce_mean(ar_loss.detach())
                    log_jepa = reduce_mean(jepa_loss.detach())
                    log_cosine = reduce_mean(jepa_metrics["cosine_sim"])
                    log_source_norm = reduce_mean(jepa_metrics["source_norm"])
                    log_target_norm = reduce_mean(jepa_metrics["target_norm"])
                    log_source_std = reduce_mean(jepa_metrics["source_std"])
                    log_target_std = reduce_mean(jepa_metrics["target_std"])
                    log_pairs = reduce_sum(jepa_metrics["pairs"])
                    avg_step_seconds = total_update_seconds / global_step
                    eta = format_eta(avg_step_seconds * (total_steps - global_step))
                    rank0_print(
                        f"epoch={epoch + 1} step={global_step} loss={log_loss.item():.4f} "
                        f"ar={log_ar.item():.4f} jepa={log_jepa.item():.4f} "
                        f"cos={log_cosine.item():.4f} pairs={int(log_pairs.item())} "
                        f"source_norm={log_source_norm.item():.2f} target_norm={log_target_norm.item():.2f} "
                        f"source_std={log_source_std.item():.4f} target_std={log_target_std.item():.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} step_time={step_seconds:.2f}s "
                        f"avg_step_time={avg_step_seconds:.2f}s eta={eta}"
                    )
        if global_step >= total_steps:
            break
        rank0_print(f"epoch={epoch + 1} done seconds={time.time() - t0:.1f}")
        if args.save_every_epoch:
            save_checkpoint(args, model, tokenizer, f"epoch_{epoch + 1}")

    save_checkpoint(args, model, tokenizer, "final")
    cleanup_distributed()


if __name__ == "__main__":
    main()
