#!/usr/bin/env python
"""Small local checks for mask and JEPA pair construction."""

from __future__ import annotations

import argparse

import torch

from med_jepa_common import pack_events
from train_jepa import gather_jepa_pairs
import build_eval_data


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(text) % 50 + 1]


def test_pack_events() -> None:
    rows = pack_events(1, [[10, 11, 99], [20, 21, 99]], seq_len=8, pad_token_id=0)
    assert len(rows) == 1
    row = rows[0]
    assert row["input_ids"] == [10, 11, 99, 20, 21, 99, 0, 0]
    assert row["event_ids"] == [0, 0, 0, 1, 1, 1, -1, -1]
    assert row["num_events"] == 2


def test_jepa_pair_target() -> None:
    input_ids = torch.tensor([[10, 99, 20, 21, 99, 30, 99]])
    attention_mask = torch.ones_like(input_ids)
    event_ids = torch.tensor([[0, 0, 1, 1, 1, 2, 2]])
    hidden = torch.arange(7 * 4, dtype=torch.float32).view(1, 7, 4)
    source, target = gather_jepa_pairs(hidden, hidden, input_ids, attention_mask, event_ids, 99, future_k=1)
    assert source is not None and target is not None
    assert source.shape[0] == 2
    assert torch.equal(source[0], hidden[0, 1])
    assert torch.equal(target[0], hidden[0, [2, 3]].mean(dim=0))
    assert torch.equal(source[1], hidden[0, 4])
    assert torch.equal(target[1], hidden[0, [5]].mean(dim=0))


def test_ethos_style_icu_mortality_labels() -> None:
    build_eval_data._TOKENIZER = FakeTokenizer()
    build_eval_data._ARGS = argparse.Namespace(seq_len=16, task="icu_mortality")
    events = [
        build_eval_data.TimelineEvent(1, None, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", 10, None, "hosp", "0", 0),
        build_eval_data.TimelineEvent(1, None, "ICU_ADMISSION//Medical Intensive Care Unit (MICU)", 10, 20, "icu_adm", "1", 1),
        build_eval_data.TimelineEvent(1, None, "LAB//50862//g/dL", 10, 20, "lab", "2", 2),
        build_eval_data.TimelineEvent(1, None, "MEDS_DEATH", 10, 20, "death", "3", 3),
        build_eval_data.TimelineEvent(1, None, "ICU_DISCHARGE//Medical Intensive Care Unit (MICU)", 10, 20, "icu_dc", "4", 4),
    ]
    samples = build_eval_data._icu_mortality_samples(1, events)
    assert len(samples) == 1
    assert samples[0]["label"] == 1
    assert samples[0]["outcome_code"] == "MEDS_DEATH"
    assert samples[0]["true_token_dist"] == 2


def main() -> None:
    test_pack_events()
    test_jepa_pair_target()
    test_ethos_style_icu_mortality_labels()
    print("smoke tests passed")


if __name__ == "__main__":
    main()
