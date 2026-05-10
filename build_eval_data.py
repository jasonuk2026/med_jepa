#!/usr/bin/env python
"""Build ETHOS-style labeled evaluation data from MEDS timelines.

This script intentionally does not read external task-label parquet files.  It
recreates the downstream task sample definition from ETHOS datasets on the raw
MEDS event stream, then tokenizes the selected history with the Qwen tokenizer.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from med_jepa_common import (
    build_mimic_description_maps,
    format_event_text,
    normalize_optional_int,
    normalize_optional_str,
    pack_events,
    parse_meds_event,
    shard_paths,
    unique_event_key,
    write_json,
)


@dataclass(frozen=True)
class TimelineEvent:
    subject_id: int
    time: Any
    code: str
    hadm_id: int | None
    stay_id: int | None
    text: str
    key: str
    order: int


_DESC_MAPS: dict[str, dict[str, str]] | None = None
_TOKENIZER = None
_ARGS: argparse.Namespace | None = None


def _init_worker(args_dict: dict[str, Any], desc_maps: dict[str, dict[str, str]]) -> None:
    global _DESC_MAPS, _TOKENIZER, _ARGS
    if _ARGS is None:
        _ARGS = argparse.Namespace(**args_dict)
    if _DESC_MAPS is None:
        _DESC_MAPS = desc_maps
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(_ARGS.model_name, trust_remote_code=True)
        if _TOKENIZER.pad_token_id is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token


def _event_sort_key(event: TimelineEvent) -> tuple[int, pd.Timestamp, int, str]:
    ts = pd.Timestamp.min if event.time is None or pd.isna(event.time) else pd.Timestamp(event.time)
    return event.subject_id, ts, event.order, event.key


def _parse_shard_events(shard_idx: int, path: str) -> dict[int, list[TimelineEvent]]:
    assert _DESC_MAPS is not None
    df = pd.read_parquet(path)
    events_by_subject: dict[int, list[TimelineEvent]] = {}
    for row_idx, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        subject_id = int(row_dict["subject_id"])
        raw_code = normalize_optional_str(row_dict.get("code")) or "UNKNOWN"
        table, code, desc, value, unit = parse_meds_event(
            raw_code,
            row_dict.get("numeric_value"),
            row_dict.get("text_value"),
            _DESC_MAPS,
        )
        time = row_dict.get("time")
        hadm_id = normalize_optional_int(row_dict.get("hadm_id"))
        stay_id = normalize_optional_int(row_dict.get("icustay_id"))
        text = format_event_text(table, code, desc, value, unit)
        key = unique_event_key(subject_id, time, table, code, value, desc)
        events_by_subject.setdefault(subject_id, []).append(
            TimelineEvent(
                subject_id=subject_id,
                time=time,
                code=raw_code,
                hadm_id=hadm_id,
                stay_id=stay_id,
                text=text,
                key=key,
                order=shard_idx * 10_000_000_000 + row_idx,
            )
        )
    for events in events_by_subject.values():
        events.sort(key=_event_sort_key)
    return events_by_subject


def _is_code(code: str, base: str) -> bool:
    return code == base or code.startswith(base + "//")


def _first_outcome_after(events: list[TimelineEvent], start_idx: int, outcome_bases: tuple[str, ...]) -> tuple[int, TimelineEvent] | None:
    for idx in range(start_idx + 1, len(events)):
        if any(_is_code(events[idx].code, base) for base in outcome_bases):
            return idx, events[idx]
    return None


def _tokenize_history(events: list[TimelineEvent]) -> list[list[int]]:
    assert _TOKENIZER is not None
    tokenized = []
    for event in events:
        ids = _TOKENIZER.encode(event.text, add_special_tokens=False)
        ids.append(_TOKENIZER.eos_token_id)
        tokenized.append(ids)
    return tokenized


def _make_sample(
    subject_id: int,
    history: list[TimelineEvent],
    label: int,
    prediction_event: TimelineEvent,
    outcome_event: TimelineEvent,
    outcome_idx: int,
    prediction_idx: int,
) -> dict[str, Any] | None:
    assert _TOKENIZER is not None and _ARGS is not None
    if not history:
        return None
    chunks = pack_events(subject_id, _tokenize_history(history), _ARGS.seq_len, _TOKENIZER.pad_token_id)
    if not chunks:
        return None
    sample = chunks[-1]
    pred_time = None if prediction_event.time is None or pd.isna(prediction_event.time) else pd.Timestamp(prediction_event.time)
    out_time = None if outcome_event.time is None or pd.isna(outcome_event.time) else pd.Timestamp(outcome_event.time)
    true_time_us = -1
    if pred_time is not None and out_time is not None:
        true_time_us = int((out_time - pred_time).total_seconds() * 1_000_000)
    sample.update(
        {
            "hadm_id": prediction_event.hadm_id if prediction_event.hadm_id is not None else -1,
            "stay_id": prediction_event.stay_id if prediction_event.stay_id is not None else -1,
            "label": label,
            "prediction_time": pred_time,
            "outcome_time": out_time,
            "outcome_code": outcome_event.code,
            "true_token_dist": outcome_idx - prediction_idx,
            "true_token_time_us": true_time_us,
        }
    )
    return sample


def _samples_for_subject(item: tuple[int, list[TimelineEvent]]) -> list[dict[str, Any]]:
    subject_id, events = item
    if _ARGS is None:
        raise RuntimeError("worker not initialized")
    if _ARGS.task == "icu_mortality":
        return _icu_mortality_samples(subject_id, events)
    if _ARGS.task == "hospital_mortality":
        return _hospital_mortality_samples(subject_id, events)
    raise ValueError(f"Unsupported task: {_ARGS.task}")


def _icu_mortality_samples(subject_id: int, events: list[TimelineEvent]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for pred_idx, event in enumerate(events):
        if not _is_code(event.code, "ICU_ADMISSION"):
            continue
        outcome = _first_outcome_after(events, pred_idx, ("ICU_DISCHARGE", "MEDS_DEATH"))
        if outcome is None:
            continue
        outcome_idx, outcome_event = outcome
        label = int(_is_code(outcome_event.code, "MEDS_DEATH"))
        sample = _make_sample(subject_id, events[: pred_idx + 1], label, event, outcome_event, outcome_idx, pred_idx)
        if sample is not None:
            samples.append(sample)
    return samples


def _hospital_mortality_samples(subject_id: int, events: list[TimelineEvent]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for adm_idx, event in enumerate(events):
        if not _is_code(event.code, "HOSPITAL_ADMISSION"):
            continue
        outcome = _first_outcome_after(events, adm_idx, ("HOSPITAL_DISCHARGE", "MEDS_DEATH"))
        if outcome is None:
            continue
        outcome_idx, outcome_event = outcome
        label = int(_is_code(outcome_event.code, "MEDS_DEATH"))
        sample = _make_sample(subject_id, events[: adm_idx + 1], label, event, outcome_event, outcome_idx, adm_idx)
        if sample is not None:
            samples.append(sample)
    return samples


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("subject_id", pa.int64()),
            ("hadm_id", pa.int64()),
            ("stay_id", pa.int64()),
            ("label", pa.int64()),
            ("prediction_time", pa.timestamp("us")),
            ("outcome_time", pa.timestamp("us")),
            ("outcome_code", pa.string()),
            ("true_token_dist", pa.int32()),
            ("true_token_time_us", pa.int64()),
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


def _process_shard_part(item: tuple[str, int, str]) -> dict[str, Any]:
    assert _ARGS is not None
    split, shard_idx, path = item
    output_dir = Path(_ARGS.output_dir)
    parts_dir = output_dir / _ARGS.task / f".{split}.parquet.parts"
    part_path = parts_dir / f"part-{shard_idx:05d}.parquet"
    print(f"split={split} start_shard={shard_idx} path={path} output={part_path}", flush=True)

    events_by_subject = _parse_shard_events(shard_idx, path)
    items = sorted(events_by_subject.items())
    if _ARGS.max_subjects:
        items = items[: _ARGS.max_subjects]

    total_rows = 0
    positives = 0
    buffer: list[dict[str, Any]] = []
    with pq.ParquetWriter(part_path, _schema(), compression="zstd") as writer:
        for subject_idx, item in enumerate(items, start=1):
            subject_samples = _samples_for_subject(item)
            if _ARGS.max_samples and total_rows + len(buffer) + len(subject_samples) > _ARGS.max_samples:
                keep = max(0, _ARGS.max_samples - total_rows - len(buffer))
                subject_samples = subject_samples[:keep]
            positives += sum(int(row["label"]) for row in subject_samples)
            buffer.extend(subject_samples)
            if len(buffer) >= _ARGS.flush_rows:
                _write_rows(writer, buffer)
                total_rows += len(buffer)
                buffer.clear()
            if _ARGS.max_samples and total_rows + len(buffer) >= _ARGS.max_samples:
                break
            if subject_idx % 1000 == 0:
                print(f"split={split} shard={shard_idx} subjects={subject_idx}/{len(items)} rows={total_rows + len(buffer)}", flush=True)
        if buffer:
            _write_rows(writer, buffer)
            total_rows += len(buffer)

    print(f"split={split} done_shard={shard_idx} rows={total_rows} positives={positives} output={part_path}", flush=True)
    return {"path": str(part_path), "rows": total_rows, "positives": positives, "subjects": len(items)}


def _merge_parts(part_paths: list[Path], output_path: Path) -> int:
    total_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--meds_dir", type=Path, required=True)
    parser.add_argument("--mimic_raw_dir", type=Path, required=True)
    parser.add_argument("--task", choices=["icu_mortality", "hospital_mortality"], default="icu_mortality")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--flush_rows", type=int, default=2048)
    return parser.parse_args()


def _build_split(args: argparse.Namespace, split: str, desc_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    global _ARGS, _DESC_MAPS, _TOKENIZER
    paths = shard_paths(args.meds_dir, split)
    if args.max_shards:
        paths = paths[: args.max_shards]
    if not paths:
        raise FileNotFoundError(f"No MEDS shards found under {args.meds_dir / split}")

    split_dir = args.output_dir / args.task
    parts_dir = split_dir / f".{split}.parquet.parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    args_dict = vars(args).copy()
    for key in ("meds_dir", "mimic_raw_dir", "output_dir"):
        args_dict[key] = str(args_dict[key])
    _ARGS = args
    _DESC_MAPS = desc_maps
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if _TOKENIZER.pad_token_id is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token

    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(ctx_name)
    indexed_paths = [(split, idx, str(path)) for idx, path in enumerate(paths)]
    part_results: list[dict[str, Any]] = []
    with ctx.Pool(args.num_workers, initializer=_init_worker, initargs=(args_dict, desc_maps)) as pool:
        for shard_count, result in enumerate(pool.imap_unordered(_process_shard_part, indexed_paths), start=1):
            part_results.append(result)
            rows_so_far = sum(int(part["rows"]) for part in part_results)
            print(f"split={split} processed_shards={shard_count}/{len(paths)} part_rows={rows_so_far}", flush=True)

    out_path = args.output_dir / args.task / f"{split}.parquet"
    part_paths = sorted(Path(part["path"]) for part in part_results)
    rows = _merge_parts(part_paths, out_path)
    shutil.rmtree(parts_dir)
    positives = sum(int(part["positives"]) for part in part_results)
    subjects = sum(int(part["subjects"]) for part in part_results)
    print(f"wrote split={split} rows={rows} positives={positives} to {out_path}", flush=True)
    return {
        "subjects": subjects,
        "rows": rows,
        "positives": positives,
        "positive_rate": positives / rows if rows else None,
        "output": str(out_path),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    desc_maps = build_mimic_description_maps(args.mimic_raw_dir)
    metadata: dict[str, Any] = {
        "task": args.task,
        "label_source": "ethos_style_meds_timeline",
        "seq_len": args.seq_len,
        "splits": {},
        "definition": {
            "icu_mortality": "history through ICU_ADMISSION; positive if first subsequent ICU_DISCHARGE/MEDS_DEATH outcome is MEDS_DEATH",
            "hospital_mortality": "history through HOSPITAL_ADMISSION; positive if first subsequent HOSPITAL_DISCHARGE/MEDS_DEATH outcome is MEDS_DEATH",
        }[args.task],
    }
    for split in args.splits:
        metadata["splits"][split] = _build_split(args, split, desc_maps)
    write_json(args.output_dir / args.task / "metadata.json", metadata)


if __name__ == "__main__":
    main()
