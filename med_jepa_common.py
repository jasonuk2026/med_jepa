"""Shared utilities for MEDS event JEPA experiments."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


DEFAULT_EVENT_TOKENS_PATH = Path(
    "/lus/lfs1aip2/scratch/u6dk/zduan.u6dk/codes/ehr/hx1/qwen3_0.6b_patient_events.parquet"
)
@dataclass(frozen=True)
class EventRecord:
    subject_id: int
    time: Any
    hadm_id: int | None
    stay_id: int | None
    text: str
    key: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def dist_rank() -> int:
    return torch.distributed.get_rank() if is_dist() else 0


def dist_world_size() -> int:
    return torch.distributed.get_world_size() if is_dist() else 1


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return rank, world, local_rank


def cleanup_distributed() -> None:
    if is_dist():
        torch.distributed.destroy_process_group()


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if dist_rank() == 0:
        print(*args, **kwargs, flush=True)


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_mimic_description_maps(mimic_raw_dir: str | Path) -> dict[str, dict[str, str]]:
    mimic_raw_dir = Path(mimic_raw_dir)
    maps: dict[str, dict[str, str]] = {"loinc": {}, "icd_diag": {}, "icd_proc": {}}

    lab_path = mimic_raw_dir / "hosp" / "d_labitems.csv.gz"
    if lab_path.exists():
        labs = pd.read_csv(lab_path, dtype={"itemid": str})
        for _, row in labs.iterrows():
            label = normalize_optional_str(row.get("label"))
            fluid = normalize_optional_str(row.get("fluid"))
            category = normalize_optional_str(row.get("category"))
            desc = " | ".join(x for x in (label, fluid, category) if x)
            if desc:
                maps["loinc"][str(row["itemid"])] = desc

    diag_path = mimic_raw_dir / "hosp" / "d_icd_diagnoses.csv.gz"
    if diag_path.exists():
        diag = pd.read_csv(diag_path, dtype={"icd_code": str, "icd_version": str})
        for _, row in diag.iterrows():
            title = normalize_optional_str(row.get("long_title"))
            if title:
                maps["icd_diag"][f"{row['icd_version']}:{row['icd_code']}"] = title

    proc_path = mimic_raw_dir / "hosp" / "d_icd_procedures.csv.gz"
    if proc_path.exists():
        proc = pd.read_csv(proc_path, dtype={"icd_code": str, "icd_version": str})
        for _, row in proc.iterrows():
            title = normalize_optional_str(row.get("long_title"))
            if title:
                maps["icd_proc"][f"{row['icd_version']}:{row['icd_code']}"] = title

    return maps


def parse_meds_event(
    code: Any,
    numeric_value: Any = None,
    text_value: Any = None,
    desc_maps: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str, str | None, str | None, str | None]:
    desc_maps = desc_maps or {}
    code_str = normalize_optional_str(code) or "UNKNOWN"
    text = normalize_optional_str(text_value)
    value = normalize_optional_str(numeric_value)
    unit = None

    if code_str.startswith("LOINC/"):
        itemid = code_str.split("/", 1)[1]
        return "labevents", itemid, desc_maps.get("loinc", {}).get(itemid), value, unit
    if code_str.startswith("LAB//"):
        parts = code_str.split("//")
        itemid = parts[1] if len(parts) > 1 else code_str
        unit = parts[2] if len(parts) > 2 and parts[2] != "UNK" else None
        return "labevents", itemid, desc_maps.get("loinc", {}).get(itemid), value, unit
    if code_str.startswith("ICD10CM/"):
        icd = code_str.split("/", 1)[1]
        return "diagnoses_icd", icd, desc_maps.get("icd_diag", {}).get(f"10:{icd}"), value, unit
    if code_str.startswith("ICD9CM/"):
        icd = code_str.split("/", 1)[1]
        return "diagnoses_icd", icd, desc_maps.get("icd_diag", {}).get(f"9:{icd}"), value, unit
    if code_str.startswith("ICD10PCS/"):
        icd = code_str.split("/", 1)[1]
        return "procedures_icd", icd, desc_maps.get("icd_proc", {}).get(f"10:{icd}"), value, unit
    if code_str.startswith("ICD9Proc/"):
        icd = code_str.split("/", 1)[1]
        return "procedures_icd", icd, desc_maps.get("icd_proc", {}).get(f"9:{icd}"), value, unit
    if code_str.startswith("MED/"):
        return "prescriptions", code_str.split("/", 1)[1], text, value, unit
    if code_str.startswith("DRG/"):
        return "drgcodes", code_str.split("/", 1)[1], text, value, unit
    if code_str.startswith("HCPCS/"):
        return "hcpcsevents", code_str.split("/", 1)[1], text, value, unit
    if code_str.startswith("MIMIC-IV/"):
        parts = code_str.split("/")
        table = parts[1].lower() if len(parts) > 1 else "mimic"
        norm = "/".join(parts[2:]) if len(parts) > 2 else code_str
        return table, norm, text, value, unit
    return "meds", code_str, text, value, unit


def format_event_text(
    table: str,
    code: str,
    desc: str | None,
    value: str | None,
    unit: str | None,
) -> str:
    parts = [f"table={table}", f"code={code}"]
    if desc:
        parts.append(f"desc={desc}")
    if value:
        parts.append(f"value={value}")
    if unit:
        parts.append(f"unit={unit}")
    return " ; ".join(parts)


def unique_event_key(subject_id: int, time: Any, table: str, code: str, value: str | None, text: str | None) -> str:
    ts = "NA" if time is None or pd.isna(time) else pd.Timestamp(time).isoformat()
    return f"{subject_id}|{ts}|{table}|{code}|{value or ''}|{text or ''}"


def dataframe_to_events(df: pd.DataFrame, desc_maps: dict[str, dict[str, str]]) -> list[EventRecord]:
    events: list[EventRecord] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        subject_id = int(row_dict["subject_id"])
        table, code, desc, value, unit = parse_meds_event(
            row_dict.get("code"),
            row_dict.get("numeric_value"),
            row_dict.get("text_value"),
            desc_maps,
        )
        time = row_dict.get("time")
        hadm_id = normalize_optional_int(row_dict.get("hadm_id"))
        stay_id = normalize_optional_int(row_dict.get("icustay_id"))
        text = format_event_text(table, code, desc, value, unit)
        key = unique_event_key(subject_id, time, table, code, value, desc)
        events.append(EventRecord(subject_id, time, hadm_id, stay_id, text, key))
    events.sort(key=lambda e: (e.subject_id, pd.Timestamp.min if e.time is None or pd.isna(e.time) else pd.Timestamp(e.time), e.key))
    return events


def pack_events(
    subject_id: int,
    event_token_ids: list[list[int]],
    seq_len: int,
    pad_token_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur_ids: list[int] = []
    cur_event_ids: list[int] = []
    chunk_idx = 0
    event_idx = 0

    def flush() -> None:
        nonlocal chunk_idx, cur_ids, cur_event_ids
        if not cur_ids:
            return
        n = len(cur_ids)
        rows.append(
            {
                "subject_id": subject_id,
                "chunk_idx": chunk_idx,
                "input_ids": cur_ids + [pad_token_id] * (seq_len - n),
                "attention_mask": [1] * n + [0] * (seq_len - n),
                "event_ids": cur_event_ids + [-1] * (seq_len - n),
                "num_valid_tokens": n,
                "num_events": max(cur_event_ids) + 1 if cur_event_ids else 0,
            }
        )
        chunk_idx += 1
        cur_ids = []
        cur_event_ids = []

    for ids in event_token_ids:
        if not ids:
            continue
        if len(ids) > seq_len:
            ids = ids[-seq_len:]
        if cur_ids and len(cur_ids) + len(ids) > seq_len:
            flush()
            event_idx = 0
        cur_ids.extend(ids)
        cur_event_ids.extend([event_idx] * len(ids))
        event_idx += 1
    flush()
    return rows


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def shard_paths(meds_dir: str | Path, split: str) -> list[Path]:
    split_dir = Path(meds_dir) / split
    return sorted(split_dir.glob("*.parquet"))


def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
