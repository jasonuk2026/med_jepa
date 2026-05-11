#!/usr/bin/env python
"""Build event-packed parquet data for Event-JEPA pretraining from MEDS shards."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from med_jepa_common import (
    build_mimic_description_maps,
    format_event_text,
    parse_meds_event,
    pack_events,
    shard_paths,
    write_json,
)

_TOKENIZER = None
_DESC_MAPS: dict[str, dict[str, str]] | None = None
_ARGS: argparse.Namespace | None = None


def _process_shard(item: tuple[int, str]) -> dict[str, Any]:
    assert _TOKENIZER is not None and _DESC_MAPS is not None and _ARGS is not None
    shard_idx, path = item
    output_path = Path(_ARGS.output_path)
    parts_dir = output_path.with_name(f".{output_path.name}.parts")
    part_path = parts_dir / f"part-{shard_idx:05d}.parquet"
    print(f"start_shard={shard_idx} path={path} output={part_path}", flush=True)

    read_columns = ["subject_id", "time", "code", "numeric_value", "text_value"]
    df = pd.read_parquet(path, columns=read_columns)
    if _ARGS.max_rows_per_shard:
        df = df.head(_ARGS.max_rows_per_shard)
    df = df.sort_values(["subject_id", "time", "code"], kind="mergesort")

    selected_subjects: set[int] | None = None
    if _ARGS.max_patients_per_shard:
        selected_subjects = set(df["subject_id"].drop_duplicates().head(_ARGS.max_patients_per_shard).astype(int))

    total_rows = 0
    buffer: list[dict[str, Any]] = []

    def write_subject(subject_id: int, subject_token_ids: list[list[int]], writer: pq.ParquetWriter) -> None:
        nonlocal total_rows, buffer
        buffer.extend(pack_events(subject_id, subject_token_ids, _ARGS.seq_len, _TOKENIZER.pad_token_id))
        if len(buffer) >= _ARGS.flush_rows:
            _write_rows(writer, buffer)
            total_rows += len(buffer)
            buffer.clear()

    with pq.ParquetWriter(part_path, _schema(), compression="zstd") as writer:
        current_subject_id: int | None = None
        subject_token_ids: list[list[int]] = []
        for row in df.itertuples(index=False):
            subject_id = int(row.subject_id)
            if selected_subjects is not None and subject_id not in selected_subjects:
                continue
            if current_subject_id is not None and subject_id != current_subject_id:
                write_subject(current_subject_id, subject_token_ids, writer)
                subject_token_ids = []
            current_subject_id = subject_id
            table, code, desc, value, unit = parse_meds_event(
                row.code,
                row.numeric_value,
                row.text_value,
                _DESC_MAPS,
            )
            text = format_event_text(table, code, desc, value, unit)
            token_ids = _TOKENIZER.encode(text, add_special_tokens=False)
            if not _ARGS.no_eot:
                token_ids.append(_TOKENIZER.eos_token_id)
            subject_token_ids.append(token_ids)
        if current_subject_id is not None:
            write_subject(current_subject_id, subject_token_ids, writer)
        if buffer:
            _write_rows(writer, buffer)
            total_rows += len(buffer)
    print(f"done_shard={shard_idx} rows={total_rows} output={part_path}", flush=True)
    return {"path": str(part_path), "rows": total_rows}


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
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=_schema()))


def _merge_parts(part_paths: list[Path], output_path: Path) -> int:
    total_rows = 0
    with pq.ParquetWriter(output_path, _schema(), compression="zstd") as writer:
        for part_path in part_paths:
            parquet_file = pq.ParquetFile(part_path)
            for batch in parquet_file.iter_batches(batch_size=2048):
                table = pa.Table.from_batches([batch], schema=_schema())
                writer.write_table(table)
                total_rows += table.num_rows
    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meds_dir", type=Path, required=True, help="Coherent MEDS data dir containing split subdirs, e.g. .../mimic-2.2-meds/data")
    parser.add_argument("--mimic_raw_dir", type=Path, required=True, help="Matching raw MIMIC dir, e.g. .../mimic-iv-2.2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_path", type=Path, default=Path("data/pretrain/train.parquet"))
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--max_rows_per_shard", type=int, default=0)
    parser.add_argument("--max_patients_per_shard", type=int, default=0)
    parser.add_argument("--flush_rows", type=int, default=2048)
    parser.add_argument("--no_eot", action="store_true", help="Do not append the tokenizer EOS/EOT token after each event.")
    return parser.parse_args()


def main() -> None:
    global _TOKENIZER, _DESC_MAPS, _ARGS
    args = parse_args()
    if not args.meds_dir.exists():
        raise FileNotFoundError(f"MEDS directory not found: {args.meds_dir}")
    if not args.mimic_raw_dir.exists():
        raise FileNotFoundError(f"MIMIC raw directory not found: {args.mimic_raw_dir}")

    paths = shard_paths(args.meds_dir, args.split)
    if args.max_shards:
        paths = paths[: args.max_shards]
    if not paths:
        raise FileNotFoundError(f"No parquet shards found under {args.meds_dir / args.split}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_path.with_name(f".{args.output_path.name}.parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    desc_maps = build_mimic_description_maps(args.mimic_raw_dir)
    _ARGS = args
    _DESC_MAPS = desc_maps
    _TOKENIZER = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    ctx = mp.get_context("fork")
    part_results: list[dict[str, Any]] = []
    with ctx.Pool(processes=args.num_workers) as pool:
        indexed_paths = [(idx, str(path)) for idx, path in enumerate(paths)]
        for processed_count, result in enumerate(pool.imap_unordered(_process_shard, indexed_paths), start=1):
            part_results.append(result)
            total_part_rows = sum(int(part["rows"]) for part in part_results)
            print(f"processed_shards={processed_count}/{len(paths)} part_rows={total_part_rows}", flush=True)

    part_paths = sorted(Path(part["path"]) for part in part_results)
    total_rows = _merge_parts(part_paths, args.output_path)
    shutil.rmtree(parts_dir)

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
            "append_eot": not args.no_eot,
            "eot_token": None if args.no_eot else "<|endoftext|>",
            "source": "meds_raw_rebuilt",
        },
    )
    print(f"wrote {total_rows} rows to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
