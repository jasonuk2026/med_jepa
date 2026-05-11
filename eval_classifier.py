#!/usr/bin/env python
"""Frozen-backbone classifier evaluation for packed EHR event data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer

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


class EvalDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.table = pq.read_table(path)

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.table.slice(idx, 1).to_pydict()
        return {
            "input_ids": torch.tensor(row["input_ids"][0], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"][0], dtype=torch.long),
            "event_ids": torch.tensor(row["event_ids"][0], dtype=torch.long),
            "label": torch.tensor(row["label"][0], dtype=torch.float32),
        }


def make_model_attention_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, eot_token_id: int, eot_attention: str) -> torch.Tensor:
    if eot_attention == "none":
        return attention_mask
    keep = attention_mask.bool() & (input_ids != eot_token_id)
    if eot_attention == "keep_last":
        valid_eot = attention_mask.bool() & (input_ids == eot_token_id)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        last_eot_idx = torch.where(valid_eot, positions, torch.zeros_like(positions)).max(dim=1, keepdim=True).values
        last_eot = valid_eot & (positions == last_eot_idx) & valid_eot.any(dim=1, keepdim=True)
        keep = keep | last_eot
    fallback = attention_mask.bool()
    keep = torch.where(keep.any(dim=1, keepdim=True), keep, fallback)
    return keep.to(dtype=attention_mask.dtype)


def append_sequence_token(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    event_ids: torch.Tensor,
    append_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if append_token_id is None:
        return input_ids, attention_mask, event_ids

    batch, seq_len = input_ids.shape
    new_input_ids = input_ids.clone()
    new_attention_mask = attention_mask.clone()
    new_event_ids = event_ids.clone()
    for row_idx in range(batch):
        valid_len = int(attention_mask[row_idx].sum().item())
        if valid_len >= seq_len:
            new_input_ids[row_idx, :-1] = input_ids[row_idx, 1:]
            new_attention_mask[row_idx, :-1] = attention_mask[row_idx, 1:]
            new_event_ids[row_idx, :-1] = event_ids[row_idx, 1:]
            insert_idx = seq_len - 1
        else:
            insert_idx = valid_len
        new_input_ids[row_idx, insert_idx] = append_token_id
        new_attention_mask[row_idx, insert_idx] = 1
        new_event_ids[row_idx, insert_idx] = -1
    return new_input_ids, new_attention_mask, new_event_ids


def pool_hidden(hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, eot_token_id: int, pooling: str) -> torch.Tensor:
    valid = attention_mask.bool()
    if pooling == "mean_all":
        return (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
    if pooling in {"last_token", "appended_token"}:
        idx = valid.long().sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    non_eot = valid & (input_ids != eot_token_id)
    if pooling == "last_non_eot":
        fallback = valid.long().sum(dim=1).clamp_min(1) - 1
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        idx = torch.where(non_eot, positions, torch.zeros_like(positions)).max(dim=1).values
        idx = torch.where(non_eot.any(dim=1), idx, fallback)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    if pooling == "mean_non_eot":
        denom = non_eot.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * non_eot.unsqueeze(-1)).sum(dim=1) / denom
    eot = valid & (input_ids == eot_token_id)
    if pooling == "mean_eot":
        denom = eot.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * eot.unsqueeze(-1)).sum(dim=1) / denom
    if pooling == "last_eot":
        fallback = valid.long().sum(dim=1).clamp_min(1) - 1
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        idx = torch.where(eot, positions, torch.zeros_like(positions)).max(dim=1).values
        idx = torch.where(eot.any(dim=1), idx, fallback)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    raise ValueError(f"unknown pooling: {pooling}")


def token_to_id(tokenizer, token: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    if token_id == tokenizer.unk_token_id and token != tokenizer.unk_token:
        return None
    return int(token_id)


def infer_eot_token_id(eval_parquet_dir: Path, task: str, tokenizer, override_token: str | None) -> int:
    if override_token:
        token_id = token_to_id(tokenizer, override_token)
        if token_id is None:
            raise ValueError(f"Could not resolve --eot_token {override_token!r} with this tokenizer")
        return token_id

    candidates: list[int] = []
    for token_id in (
        tokenizer.eos_token_id,
        token_to_id(tokenizer, "<|im_end|>"),
        token_to_id(tokenizer, "<|endoftext|>"),
    ):
        if token_id is not None and int(token_id) not in candidates:
            candidates.append(int(token_id))

    counts = {token_id: 0 for token_id in candidates}
    if counts:
        table = pq.read_table(
            eval_parquet_dir / task / "train.parquet",
            columns=["input_ids", "attention_mask", "event_ids"],
        ).slice(0, 256)
        rows = table.to_pydict()
        for input_ids, attention_mask, event_ids in zip(rows["input_ids"], rows["attention_mask"], rows["event_ids"]):
            for token_id, mask, event_id in zip(input_ids, attention_mask, event_ids):
                if mask and event_id >= 0 and token_id in counts:
                    counts[token_id] += 1
        best_id, best_count = max(counts.items(), key=lambda item: item[1])
        if best_count > 0:
            return best_id

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id and no EOT token could be inferred")
    return int(tokenizer.eos_token_id)


@torch.no_grad()
def evaluate(args: argparse.Namespace, model: nn.Module, head: nn.Module, loader: DataLoader, eot_token_id: int, dtype: torch.dtype, device: torch.device) -> dict[str, float]:
    model.eval()
    head.eval()
    preds: list[float] = []
    labels: list[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        event_ids = batch["event_ids"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        input_ids, attention_mask, event_ids = append_sequence_token(input_ids, attention_mask, event_ids, args.append_token_id)
        model_attention_mask = make_model_attention_mask(input_ids, attention_mask, eot_token_id, args.eot_attention)
        out = model(input_ids=input_ids, attention_mask=model_attention_mask, use_cache=False, return_dict=True)
        pooled = pool_hidden(out.last_hidden_state, input_ids, attention_mask, eot_token_id, args.pooling)
        prob = torch.sigmoid(head(pooled).squeeze(-1))
        preds.extend(prob.float().cpu().tolist())
        labels.extend(label.float().cpu().tolist())
    if dist_world_size() > 1:
        gathered = [None for _ in range(dist_world_size())]
        torch.distributed.all_gather_object(gathered, (preds, labels))
        preds = [x for part, _ in gathered for x in part]
        labels = [x for _, part in gathered for x in part]
    if dist_rank() != 0:
        return {}
    y_true = np.array(labels)
    y_score = np.array(preds)
    out_metrics: dict[str, float] = {"n": float(len(y_true)), "positive_rate": float(y_true.mean())}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out_metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        out_metrics["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception as exc:
        out_metrics["metric_error"] = str(exc)  # type: ignore[assignment]
    return out_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_dir", required=True)
    parser.add_argument("--eval_parquet_dir", type=Path, required=True)
    parser.add_argument("--task", default="icu_mortality")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--pooling", choices=["last_eot", "mean_eot", "last_token", "appended_token", "mean_all", "last_non_eot", "mean_non_eot"], default="mean_eot")
    parser.add_argument("--eot_attention", choices=["none", "all", "keep_last"], default="none")
    parser.add_argument("--eot_token", default=None, help="Event boundary token. Defaults to inferring from valid tokens in eval train parquet.")
    parser.add_argument("--append_token", default=None, help="Optional token appended to each sequence before pooling, e.g. <|endoftext|> for Qwen3-Embedding.")
    parser.add_argument("--ignore_eot_attention", action="store_const", const="all", dest="eot_attention", default=argparse.SUPPRESS)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"], default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_distributed()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_token_id = infer_eot_token_id(args.eval_parquet_dir, args.task, tokenizer, args.eot_token)
    args.append_token_id = token_to_id(tokenizer, args.append_token) if args.append_token else None
    if args.append_token and args.append_token_id is None:
        raise ValueError(f"Could not resolve --append_token {args.append_token!r} with this tokenizer")
    rank0_print(f"eot_token_id={eot_token_id} append_token_id={args.append_token_id}")

    model = AutoModel.from_pretrained(
        args.pretrained_dir,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    head = nn.Linear(model.config.hidden_size, 1).to(device=device, dtype=dtype)
    if args.compile:
        model = torch.compile(model)
        head = torch.compile(head)
    if world > 1:
        head = DDP(head, device_ids=[local_rank], output_device=local_rank)

    train_ds = EvalDataset(args.eval_parquet_dir / args.task / "train.parquet")
    test_ds = EvalDataset(args.eval_parquet_dir / args.task / "test.parquet")
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if world > 1 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=train_sampler is None, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auprc = -1.0
    bad_epochs = 0
    best_metrics: dict[str, float] = {}
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.eval()
        head.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            event_ids = batch["event_ids"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            input_ids, attention_mask, event_ids = append_sequence_token(input_ids, attention_mask, event_ids, args.append_token_id)
            with torch.no_grad():
                model_attention_mask = make_model_attention_mask(input_ids, attention_mask, eot_token_id, args.eot_attention)
                out = model(input_ids=input_ids, attention_mask=model_attention_mask, use_cache=False, return_dict=True)
                pooled = pool_hidden(out.last_hidden_state, input_ids, attention_mask, eot_token_id, args.pooling)
            logits = head(pooled).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits.float(), label.float())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        metrics = evaluate(args, model, head, test_loader, eot_token_id, dtype, device)
        if dist_rank() == 0:
            score = float(metrics.get("auprc", -1.0))
            rank0_print(f"epoch={epoch + 1} metrics={metrics}")
            if score > best_auprc:
                best_auprc = score
                best_metrics = metrics
                bad_epochs = 0
                args.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(unwrap_model(head).state_dict(), args.output_dir / "best_head.pt")
            else:
                bad_epochs += 1
        if world > 1:
            stop_tensor = torch.tensor([bad_epochs >= args.early_stopping_patience], device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        else:
            should_stop = bad_epochs >= args.early_stopping_patience
        if should_stop:
            break

    if dist_rank() == 0:
        write_json(args.output_dir / "metrics.json", {"best": best_metrics, "args": vars(args)})
    cleanup_distributed()


if __name__ == "__main__":
    main()
