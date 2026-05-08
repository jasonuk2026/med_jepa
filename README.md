# med_jepa

Minimal Event-JEPA pipeline for coherent EHR event-token data.

## Design notes

- EOT token: uses the tokenizer EOS token, `<|endoftext|>` for Qwen.
- No AMP: scripts do not use `torch.autocast` or `GradScaler`.
- `--dtype bf16` loads/runs weights in bf16 directly; use `--dtype fp32` for full fp32.
- Training uses normal causal attention plus the 2D padding mask.
- Flash attention 2/3 is supported through `--attn_implementation`.
- Gradient checkpointing is intentionally not enabled.
- Pretraining data extraction reads the EHR-local `patient_id/chunk_idx/event_token_ids` parquet.
- Evaluation data extraction requires explicit coherent MEDS/raw-table paths.
- Data extraction uses `multiprocessing` process pools, not threads.

## Build pretraining data

```bash
conda run -n torch python build_pretrain_data.py \
  --event_tokens_path /lus/lfs1aip2/scratch/u6dk/zduan.u6dk/codes/ehr/hx1/qwen3_0.6b_patient_events.parquet \
  --output_path data/pretrain/train.parquet \
  --seq_len 2048 \
  --num_workers 16
```

## Train JEPA

```bash
torchrun --standalone --nproc_per_node=4 train_jepa.py \
  --model_name Qwen/Qwen3-0.6B \
  --train_parquet data/pretrain/train.parquet \
  --output_dir experiments/qwen3_0p6b_event_jepa \
  --attn_implementation flash_attention_3 \
  --batch_size 4 \
  --global_batch_size 256 \
  --epochs 1 \
  --lr 2e-5 \
  --dtype bf16 \
  --compile \
  --save_every_epoch
```

## Build evaluation data

Evaluation labels are generated from the MEDS timeline with ETHOS-style task logic. For
`icu_mortality`, each `ICU_ADMISSION*` event creates one sample; the label is positive when the
first later `ICU_DISCHARGE*`/`MEDS_DEATH` outcome is `MEDS_DEATH`.

```bash
conda run -n torch python build_eval_data.py \
  --meds_dir /path/to/coherent/mimic-2.2-meds/data \
  --mimic_raw_dir /path/to/coherent/mimic-iv-2.2 \
  --task icu_mortality \
  --output_dir data/eval \
  --seq_len 2048 \
  --num_workers 16
```

## Frozen classifier eval

```bash
conda run -n torch python eval_classifier.py \
  --pretrained_dir experiments/qwen3_0p6b_event_jepa/final \
  --eval_parquet_dir data/eval \
  --task icu_mortality \
  --output_dir experiments/classifier/icu_mortality \
  --pooling mean_eot \
  --attn_implementation flash_attention_3 \
  --dtype bf16 \
  --compile
```

## Smoke tests

```bash
conda run -n torch python -m py_compile med_jepa_common.py build_pretrain_data.py train_jepa.py build_eval_data.py eval_classifier.py smoke_test.py
conda run -n torch python smoke_test.py
```
