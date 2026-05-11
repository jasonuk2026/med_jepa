# med_jepa

Minimal Event-JEPA pipeline for coherent MIMIC MEDS event streams.

## Design notes

- EOT token: uses the tokenizer EOS token, `<|endoftext|>` for Qwen.
- No AMP: scripts do not use `torch.autocast` or `GradScaler`.
- `--dtype bf16` loads/runs weights in bf16 directly; use `--dtype fp32` for full fp32.
- Training uses normal causal attention plus the 2D padding mask.
- Flash attention 2/3 is supported through `--attn_implementation`.
- Gradient checkpointing is intentionally not enabled.
- Pretraining and evaluation data extraction require explicit coherent MEDS/raw-table paths.
- Data extraction uses `multiprocessing` process pools, not threads.

## Build pretraining data

```bash
python build_pretrain_data.py \
--meds_dir data/raw/mimic-2.2-meds/data \
--mimic_raw_dir data/raw/mimic-iv-2.2 \
--output_path data/pretrain/train.parquet \
--seq_len 2048 \
--num_workers 16
```

To submit the data build to a compute node:

```bash
./run_sm.sh -j build_pretrain_fork_parts -n 0 -c 4 -m 100G -t 24:00:00 \
python build_pretrain_data.py \
--meds_dir data/raw/mimic-2.2-meds/data \
--mimic_raw_dir data/raw/mimic-iv-2.2 \
--output_path data/pretrain/train.parquet \
--seq_len 2048 \
--num_workers 4
```

To build an AR-only pretraining set without EOT tokens between events:

```bash
./run_sm.sh -j build_pretrain_no_eot -n 1 -c 72 -m 100G -t 24:00:00 \
python build_pretrain_data.py \
--meds_dir data/raw/mimic-2.2-meds/data \
--mimic_raw_dir data/raw/mimic-iv-2.2 \
--output_path data/pretrain/train_no_eot.parquet \
--seq_len 2048 \
--num_workers 8 \
--no_eot
```

Slurm logs are written under `logs/slurm/<YYYYMMDD>/<job_name>/`, with
`*.out`, `*.err`, a `*.cmd.txt` command record, and the generated `*.job.sh`
script for each submission.

## Train JEPA

Two-GPU 2k-step test:

```bash
torchrun --standalone --nproc_per_node=2 train_jepa.py \
--model_name Qwen/Qwen3-0.6B \
--train_parquet data/pretrain/train.parquet \
--output_dir experiments/qwen3_0p6b_event_jepa_2gpu_2k \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.001 \
--var_weight 0.05 \
--var_gamma 0.02 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--max_steps 2000 \
--log_steps 1 \
--save_every_epoch
```

Four-GPU full-epoch Slurm run with cosine JEPA loss:

```bash
./run_sm.sh -j pretrain_jepa_4gpu_future2_cosine_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B \
--train_parquet data/pretrain/train.parquet \
--output_dir experiments/qwen3_0p6b_event_jepa_4gpu_future2_cosine_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_loss cosine \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0.05 \
--var_gamma 0.05 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

Four-GPU full-epoch Slurm run with MSE JEPA loss:

```bash
./run_sm.sh -j pretrain_jepa_4gpu_future2_mse_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B \
--train_parquet data/pretrain/train.parquet \
--output_dir experiments/qwen3_0p6b_event_jepa_4gpu_future2_mse_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_loss mse \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0.05 \
--var_gamma 0.05 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

Four-GPU full-epoch Slurm run with MIMIC AR-only loss, ignoring EOT tokens as
prediction targets:

```bash
./run_sm.sh -j pretrain_mimic_ar_only_4gpu_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B \
--train_parquet data/pretrain/train.parquet \
--output_dir experiments/qwen3_0p6b_mimic_ar_only_4gpu_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--jepa_weight 0 \
--var_weight 0 \
--ar_weight 1.0 \
--ar_eot_weight 0.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

Four-GPU full-epoch Slurm run with MIMIC AR-only loss on no-EOT pretraining
data:

```bash
./run_sm.sh -j pretrain_mimic_ar_only_no_eot_4gpu_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_mimic_ar_only_no_eot_4gpu_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--jepa_weight 0 \
--var_weight 0 \
--ar_weight 1.0 \
--ar_eot_weight 0.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

## Build evaluation data

Evaluation labels are generated from the MEDS timeline with ETHOS-style task logic. For
`icu_mortality`, each `ICU_ADMISSION*` event creates one sample; the label is positive when the
first later `ICU_DISCHARGE*`/`MEDS_DEATH` outcome is `MEDS_DEATH`.

```bash
python build_eval_data.py \
--meds_dir data/raw/mimic-2.2-meds/data \
--mimic_raw_dir data/raw/mimic-iv-2.2 \
--task icu_mortality \
--output_dir data/eval \
--seq_len 2048 \
--num_workers 8
```

To submit evaluation data generation to a compute node:

```bash
./run_sm.sh -j build_eval_icu_mortality -n 0 -c 8 -m 100G -t 24:00:00 \
python build_eval_data.py \
--meds_dir data/raw/mimic-2.2-meds/data \
--mimic_raw_dir data/raw/mimic-iv-2.2 \
--task icu_mortality \
--output_dir data/eval \
--seq_len 2048 \
--num_workers 4
```

## Frozen classifier eval

Raw Qwen3 linear-probe baseline, keeping only the final EOT token in the model
attention mask and training logistic regression on the final non-EOT token embedding:

```bash
./run_sm.sh -j eval_raw_qwen3_icu_last_non_eot_linear -n 1 -c 16 -m 100G -t 06:00:00 \
python eval_classifier.py \
--pretrained_dir Qwen/Qwen3-0.6B \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/raw_qwen3_icu_last_non_eot_linear \
--pooling last_non_eot \
--eot_attention keep_last \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 10 \
--lr 1e-4 \
--num_workers 2
```

Four-GPU cosine-JEPA linear-probe baseline using the mean of all EOT token
embeddings:

```bash
./run_sm.sh -j eval_jepa_cosine_full_epoch_mean_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_event_jepa_4gpu_future2_cosine_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/jepa_cosine_full_epoch_mean_eot_linear_4gpu \
--pooling mean_eot \
--eot_attention none \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU MSE-JEPA linear-probe baseline using the mean of all EOT token
embeddings:

```bash
./run_sm.sh -j eval_jepa_mse_full_epoch_mean_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_event_jepa_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/jepa_mse_full_epoch_mean_eot_linear_4gpu \
--pooling mean_eot \
--eot_attention none \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3-Embedding linear-probe baseline using the final EOT token
embedding as the sequence embedding:

```bash
./run_sm.sh -j eval_qwen3_embedding_last_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir Qwen/Qwen3-Embedding-0.6B \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/qwen3_embedding_last_eot_linear_4gpu \
--pooling last_eot \
--eot_attention none \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU MIMIC AR-only linear-probe baseline using the mean of all EOT token
embeddings:

```bash
./run_sm.sh -j eval_mimic_ar_only_mean_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_mimic_ar_only_4gpu_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/mimic_ar_only_mean_eot_linear_4gpu \
--pooling mean_eot \
--eot_attention none \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

## Smoke tests

```bash
python -m py_compile med_jepa_common.py build_pretrain_data.py train_jepa.py build_eval_data.py eval_classifier.py smoke_test.py
python smoke_test.py
```
