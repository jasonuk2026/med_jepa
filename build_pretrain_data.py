#!/usr/bin/env python
"""Build event-packed parquet data for Event-JEPA pretraining."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from med_jepa_common import (
    DEFAULT_MEDS_DIR,
    DEFAULT_MIMIC_RAW_DIR,
    build_mimic_description_maps,
    dataframe_to_events,
    pack_events,
    shard_paths,
    write_json,
)

_TOKENIZER = None
_DESC_MAPS: dict[str, dict[str, str]] | None = None
_ARGS: argparse.Namespace | None = None


def _init_worker(args_dict: dict[str, Any], desc_maps: dict[str, dict[str, str]]) -> None:
    global _TOKENIZER, _DESC_MAPS, _ARGS
    _ARGS = argparse.Namespace(**args_dict)
    _DESC_MAPS = desc_maps
    _TOKENIZER = AutoTokenizer.from_pretrained(_ARGS.model_name, trust_remote_code=True)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token


def _process_shard(path: str) -> list[dict[str, Any]]:
    assert _TOKENIZER is not None and _DESC_MAPS is not None and _ARGS is not None
    df = pd.read_parquet(path)
    if _ARGS.max_rows_per_shard:
        df = df.head(_ARGS.max_rows_per_shard)
    events = dataframe_to_events(df, _DESC_MAPS)
    rows: list[dict[str, Any]] = []
    by_subject: dict[int, list[list[int]]] = {}
    for event in events:
        token_ids = _TOKENIZER.encode(event.text, add_special_tokens=False)
        token_ids.append(_TOKENIZER.eos_token_id)
        by_subject.setdefault(event.subject_id, []).append(token_ids)
    subject_ids = sorted(by_subject)
    if _ARGS.max_patients_per_shard:
        subject_ids = subject_ids[: _ARGS.max_patients_per_shard]
    for subject_id in subject_ids:
        rows.extend(pack_events(subject_id, by_subject[subject_id], _ARGS.seq_len, _TOKENIZER.pad_token_id))
    return rows


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("subject_id", pa.int64()),
            ("chunk_idx", pa.int32()),
            ("input_ids", pa.list_(pa.int32())),
            ("attention_mask", pa.list_(pa.int8())),
            ("event_ids", pa.list_(pa.int32())),
            ("num_valid_tokens", pa.int32()),
            ("num_events", pa.int32()),
        ]
    )


def _write_rows(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=_schema())
    writer.write_table(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meds_dir", type=Path, default=DEFAULT_MEDS_DIR)
    parser.add_argument("--mimic_raw_dir", type=Path, default=DEFAULT_MIMIC_RAW_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_path", type=Path, default=Path("data/pretrain/train.parquet"))
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--max_rows_per_shard", type=int, default=0)
    parser.add_argument("--max_patients_per_shard", type=int, default=0)
    parser.add_argument("--flush_rows", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = shard_paths(args.meds_dir, args.split)
    if args.max_shards:
        paths = paths[: args.max_shards]
    if not paths:
        raise FileNotFoundError(f"No parquet shards found under {args.meds_dir / args.split}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    desc_maps = build_mimic_description_maps(args.mimic_raw_dir)
    args_dict = vars(args).copy()
    args_dict["meds_dir"] = str(args.meds_dir)
    args_dict["mimic_raw_dir"] = str(args.mimic_raw_dir)
    args_dict["output_path"] = str(args.output_path)

    ctx = mp.get_context("spawn")
    buffer: list[dict[str, Any]] = []
    total_rows = 0
    with pq.ParquetWriter(args.output_path, _schema(), compression="zstd") as writer:
        with ctx.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args_dict, desc_maps),
        ) as pool:
            for shard_idx, rows in enumerate(pool.imap_unordered(_process_shard, [str(p) for p in paths]), start=1):
                buffer.extend(rows)
                if len(buffer) >= args.flush_rows:
                    _write_rows(writer, buffer)
                    total_rows += len(buffer)
                    buffer.clear()
                print(f"processed_shards={shard_idx}/{len(paths)} buffered={len(buffer)} total_rows={total_rows}", flush=True)
        if buffer:
            _write_rows(writer, buffer)
            total_rows += len(buffer)

    write_json(
        args.output_path.with_suffix(".metadata.json"),
        {
            "rows": total_rows,
            "split": args.split,
            "seq_len": args.seq_len,
            "model_name": args.model_name,
            "meds_dir": str(args.meds_dir),
            "mimic_raw_dir": str(args.mimic_raw_dir),
            "num_workers": args.num_workers,
            "eot_token": "<|endoftext|>",
        },
    )
    print(f"wrote {total_rows} rows to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
