#!/usr/bin/env python
"""Train STP-style JEPA on no-EOT packed EHR token sequences."""

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


def ar_loss_fn(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float().contiguous()
    targets = input_ids[:, 1:].contiguous()
    valid = attention_mask[:, 1:].bool()
    flat_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1), reduction="none")
    flat_valid = valid.view(-1).to(flat_loss.dtype)
    return (flat_loss * flat_valid).sum() / flat_valid.sum().clamp_min(1.0)


def span_repr(hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Represent token span [start, end) using causal hidden-state difference."""
    if end <= start:
        return hidden.new_zeros((hidden.size(-1),))
    end_state = hidden[end - 1]
    if start <= 0:
        return end_state
    return end_state - hidden[start - 1]


def normalize_span_repr(x: torch.Tensor, length: int, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "sqrt":
        return x / math.sqrt(max(1, length))
    if mode == "linear":
        return x / max(1, length)
    raise ValueError(f"unknown length_norm: {mode}")


def sample_patch(
    length: int,
    min_length: int,
    max_length: int,
    generator: torch.Generator,
    zero_start: bool,
) -> tuple[int, int] | None:
    if length < 2:
        return None
    min_len = max(1, min_length)
    max_len = length - 1
    if max_length > 0:
        max_len = min(max_len, max_length)
    if max_len < min_len:
        return None
    patch_len = int(torch.randint(min_len, max_len + 1, (), generator=generator).item())
    if zero_start:
        start = 0
    else:
        start = int(torch.randint(0, length - patch_len + 1, (), generator=generator).item())
    end = start + patch_len
    return start, end


def stp_loss_fn(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    min_patch_length: int,
    max_patch_length: int,
    patch_times: int,
    zero_start: bool,
    length_norm: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    patch_lengths: list[float] = []
    lengths = attention_mask.sum(dim=1).tolist()
    for b, length_value in enumerate(lengths):
        length = int(length_value)
        for _ in range(patch_times):
            patch = sample_patch(length, min_patch_length, max_patch_length, generator, zero_start)
            if patch is None:
                continue
            start, end = patch
            patch_length = end - start
            rest_length = length - patch_length
            before = span_repr(hidden_states[b], 0, start)
            patch_repr = span_repr(hidden_states[b], start, end)
            after = span_repr(hidden_states[b], end, length)
            sources.append(normalize_span_repr(before + after, rest_length, length_norm))
            targets.append(normalize_span_repr(patch_repr, patch_length, length_norm))
            patch_lengths.append(float(patch_length))
    if not sources:
        zero = hidden_states.float().sum() * 0.0
        metrics = {
            "cosine_sim": zero.detach(),
            "source_norm": zero.detach(),
            "target_norm": zero.detach(),
            "source_std": zero.detach(),
            "target_std": zero.detach(),
            "patch_len": zero.detach(),
            "pairs": torch.zeros((), dtype=torch.float32, device=hidden_states.device),
        }
        return zero, metrics
    source = torch.stack(sources)
    target = torch.stack(targets)
    cosine_sim_per_pair = F.cosine_similarity(source.float(), target.float(), dim=-1)
    loss = 1.0 - cosine_sim_per_pair.mean()
    metrics = {
        "cosine_sim": cosine_sim_per_pair.detach().mean(),
        "source_norm": source.detach().float().norm(dim=-1).mean(),
        "target_norm": target.detach().float().norm(dim=-1).mean(),
        "source_std": source.detach().float().std(dim=0, unbiased=False).mean(),
        "target_std": target.detach().float().std(dim=0, unbiased=False).mean(),
        "patch_len": torch.tensor(sum(patch_lengths) / len(patch_lengths), dtype=torch.float32, device=hidden_states.device),
        "pairs": torch.tensor(float(len(sources)), dtype=torch.float32, device=hidden_states.device),
    }
    return loss, metrics


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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--stp_lambda", type=float, default=1.0)
    parser.add_argument("--patch_times", type=int, default=1)
    parser.add_argument("--min_patch_length", type=int, default=1, help="Minimum patch length.")
    parser.add_argument("--max_patch_length", type=int, default=256, help="Maximum patch length. <=0 means no cap.")
    parser.add_argument("--patch_zero_start", action="store_true", help="Always start the random patch at token 0.")
    parser.add_argument("--length_norm", choices=["none", "sqrt", "linear"], default="none")
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
    patch_generator = torch.Generator(device="cpu")
    patch_generator.manual_seed(args.seed + rank * 100003)

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
        f"max_steps={args.max_steps or 'none'} stp_lambda={args.stp_lambda} "
        f"patch_times={args.patch_times} min_patch_length={args.min_patch_length} "
        f"max_patch_length={args.max_patch_length} "
        f"patch_zero_start={args.patch_zero_start} length_norm={args.length_norm}"
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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            ar_loss = ar_loss_fn(outputs.logits, input_ids, attention_mask)
            stp_loss, stp_metrics = stp_loss_fn(
                outputs.hidden_states[-1],
                attention_mask,
                args.min_patch_length,
                args.max_patch_length,
                args.patch_times,
                args.patch_zero_start,
                args.length_norm,
                patch_generator,
            )
            loss = (ar_loss + args.stp_lambda * stp_loss) / accum_steps
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
                    log_stp = reduce_mean(stp_loss.detach())
                    log_cosine = reduce_mean(stp_metrics["cosine_sim"])
                    log_source_norm = reduce_mean(stp_metrics["source_norm"])
                    log_target_norm = reduce_mean(stp_metrics["target_norm"])
                    log_source_std = reduce_mean(stp_metrics["source_std"])
                    log_target_std = reduce_mean(stp_metrics["target_std"])
                    log_patch_len = reduce_mean(stp_metrics["patch_len"])
                    log_pairs = reduce_sum(stp_metrics["pairs"])
                    avg_step_seconds = total_update_seconds / global_step
                    eta = format_eta(avg_step_seconds * (total_steps - global_step))
                    rank0_print(
                        f"epoch={epoch + 1} step={global_step} loss={log_loss.item():.4f} "
                        f"ar={log_ar.item():.4f} stp={log_stp.item():.4f} "
                        f"cos={log_cosine.item():.4f} pairs={int(log_pairs.item())} "
                        f"patch_len={log_patch_len.item():.1f} "
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
