#!/usr/bin/env python
"""Build Event-JEPA pretraining parquet from coherent EHR event-token data."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from med_jepa_common import DEFAULT_EVENT_TOKENS_PATH, pack_events, write_json


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


def _normalize_event_ids(event_token_ids: list[list[int]], eot_token_id: int) -> list[list[int]]:
    normalized = []
    for ids in event_token_ids:
        ids = [int(x) for x in ids]
        if not ids:
            continue
        if ids[-1] != eot_token_id:
            ids.append(eot_token_id)
        normalized.append(ids)
    return normalized


def _process_records(payload: tuple[int, list[dict[str, Any]], dict[str, Any]]) -> tuple[int, int, list[dict[str, Any]]]:
    batch_idx, records, args = payload
    rows: list[dict[str, Any]] = []
    for record in records:
        subject_id = int(record["patient_id"])
        events = _normalize_event_ids(record["event_token_ids"], args["eot_token_id"])
        packed = pack_events(subject_id, events, args["seq_len"], args["pad_token_id"])
        source_chunk_idx = int(record.get("chunk_idx", 0))
        for local_idx, row in enumerate(packed):
            row["chunk_idx"] = source_chunk_idx * 1000 + local_idx
            rows.append(row)
    return batch_idx, len(records), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_tokens_path", type=Path, default=DEFAULT_EVENT_TOKENS_PATH)
    parser.add_argument("--output_path", type=Path, default=Path("data/pretrain/train.parquet"))
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--source_batch_size", type=int, default=512)
    parser.add_argument("--flush_rows", type=int, default=2048)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--eot_token_id", type=int, default=151643)
    parser.add_argument("--pad_token_id", type=int, default=151643)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.event_tokens_path.exists():
        raise FileNotFoundError(f"Event-token parquet not found: {args.event_tokens_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    worker_args = {
        "seq_len": args.seq_len,
        "eot_token_id": args.eot_token_id,
        "pad_token_id": args.pad_token_id,
    }

    def iter_payloads() -> Any:
        parquet_file = pq.ParquetFile(args.event_tokens_path)
        rows_seen_local = 0
        for batch_idx, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=args.source_batch_size,
                columns=["patient_id", "chunk_idx", "event_token_ids"],
            ),
            start=1,
        ):
            records = batch.to_pylist()
            if args.max_rows:
                remaining = args.max_rows - rows_seen_local
                if remaining <= 0:
                    break
                records = records[:remaining]
            rows_seen_local += len(records)
            if records:
                yield batch_idx, records, worker_args
            if args.max_rows and rows_seen_local >= args.max_rows:
                break

    ctx = mp.get_context("spawn")
    total_rows = 0
    source_rows = 0
    processed_batches = 0
    buffer: list[dict[str, Any]] = []
    with pq.ParquetWriter(args.output_path, _schema(), compression="zstd") as writer:
        with ctx.Pool(args.num_workers) as pool:
            for batch_idx, batch_source_rows, rows in pool.imap_unordered(_process_records, iter_payloads()):
                processed_batches += 1
                source_rows += batch_source_rows
                buffer.extend(rows)
                if len(buffer) >= args.flush_rows:
                    _write_rows(writer, buffer)
                    total_rows += len(buffer)
                    buffer.clear()
                print(
                    f"processed_batches={processed_batches} last_batch={batch_idx} "
                    f"source_rows={source_rows} buffered={len(buffer)} total_rows={total_rows}",
                    flush=True,
                )
        if buffer:
            _write_rows(writer, buffer)
            total_rows += len(buffer)

    write_json(
        args.output_path.with_suffix(".metadata.json"),
        {
            "rows": total_rows,
            "source_rows": source_rows,
            "seq_len": args.seq_len,
            "event_tokens_path": str(args.event_tokens_path),
            "num_workers": args.num_workers,
            "eot_token_id": args.eot_token_id,
            "pad_token_id": args.pad_token_id,
            "source_columns": ["patient_id", "chunk_idx", "event_token_ids"],
        },
    )
    print(f"wrote {total_rows} rows to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
