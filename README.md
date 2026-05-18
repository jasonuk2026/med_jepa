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

Experiment narrative and interpretation:

- [Event-level JEPA experiment narrative](analysis/jepa_experiment_narrative.md)
- [Evaluation result table](analysis/eval_results_table.md)
- [Evaluation result plots](analysis/eval_results_table.png), [metric line plot](analysis/eval_metrics_line.png)

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

Four-GPU full-epoch Slurm run with Qwen3 Base and MIMIC AR-only loss on no-EOT
pretraining data:

```bash
./run_sm.sh -j pretrain_base_ar_only_no_eot_4gpu_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_ar_only_no_eot_4gpu_warmup10_full_epoch \
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

Four-GPU full-epoch Slurm run with Qwen3 Base and MSE JEPA loss. The
previous `data/pretrain/train.parquet` was built with `<|im_end|>` as the
event boundary token, so pass it explicitly when training the Base model:

```bash
./run_sm.sh -j pretrain_base_jepa_4gpu_future2_mse_warmup10_full_epoch_v2 -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train.parquet \
--output_dir experiments/qwen3_0p6b_base_event_jepa_4gpu_future2_mse_warmup10_full_epoch_v2 \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--eot_token '<|im_end|>' \
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

Four-GPU full-epoch Slurm run with Qwen3 Base and MSE JEPA loss on no-EOT
pretraining data. This uses the last token of each event as the JEPA source:

`--ar_weight` is the global coefficient applied after the AR cross-entropy is
averaged. `--ar_eot_weight` only controls the relative weight of EOT targets in
AR, and is irrelevant for no-EOT pretraining data.

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_4gpu_future2_mse_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_4gpu_future2_mse_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
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

Four-GPU full-epoch Slurm run with Qwen3 Base and no-EOT last-token MSE-JEPA,
while keeping AR at full weight:

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_ar1_4gpu_future2_mse_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_ar1_4gpu_future2_mse_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_loss mse \
--ar_weight 1.0 \
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

Four-GPU full-epoch no-EOT last-token MSE-JEPA run with half JEPA weight and
full AR weight:

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_jepa0p5_ar1_4gpu_future2_mse_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa0p5_ar1_4gpu_future2_mse_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_loss mse \
--jepa_weight 0.5 \
--ar_weight 1.0 \
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

Four-GPU full-epoch no-EOT last-token MSE-JEPA run with full JEPA weight and
half AR weight:

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_jepa1_ar0p5_4gpu_future2_mse_warmup10_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa1_ar0p5_4gpu_future2_mse_warmup10_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_loss mse \
--jepa_weight 1.0 \
--ar_weight 0.5 \
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

Four-GPU full-epoch no-EOT last-token MSE-JEPA reruns with fixed AR loss
weighting and no variance regularization:

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_jepa1_ar0p6_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa1_ar0p6_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_loss mse \
--jepa_weight 1.0 \
--ar_weight 0.6 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_jepa1_ar0p3_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa1_ar0p3_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_loss mse \
--jepa_weight 1.0 \
--ar_weight 0.3 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_jepa0p6_ar1_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa0p6_ar1_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_loss mse \
--jepa_weight 0.6 \
--ar_weight 1.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_token_jepa0p3_ar1_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa0p3_ar1_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_loss mse \
--jepa_weight 0.3 \
--ar_weight 1.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

Four-GPU no-EOT JEPA runs with last-token targets. These predict the last
non-pad, non-EOT token of the exact `future_k`-th future event instead of the
mean over future event tokens:

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_to_future_last_jepa1_ar0p3_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa1_ar0p3_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_target last_token \
--jepa_loss mse \
--jepa_weight 1.0 \
--ar_weight 0.3 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_to_future_last_jepa1_ar0p6_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa1_ar0p6_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_target last_token \
--jepa_loss mse \
--jepa_weight 1.0 \
--ar_weight 0.6 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_to_future_last_jepa1_ar1_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa1_ar1_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_target last_token \
--jepa_loss mse \
--jepa_weight 1.0 \
--ar_weight 1.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_to_future_last_jepa0p6_ar1_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa0p6_ar1_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_target last_token \
--jepa_loss mse \
--jepa_weight 0.6 \
--ar_weight 1.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_jepa_no_eot_last_to_future_last_jepa0p3_ar1_var0_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa0p3_ar1_var0_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--batch_size 8 \
--global_batch_size 128 \
--future_k 2 \
--jepa_source last_token \
--jepa_target last_token \
--jepa_loss mse \
--jepa_weight 0.3 \
--ar_weight 1.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--var_weight 0 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 100 \
--save_every_epoch
```

Four-GPU chunk JEPA run on the no-EOT pretraining data. `--num_chunks 2` splits
each sample into two time-ordered chunks, uses the full sequence view for AR,
and aligns independently forwarded prefix and suffix last-token states with
cosine loss. Each sample expands to `1 + 2 * (num_chunks - 1)` views.

Single-GPU short test:

```bash
python train_chunk_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_chunk_jepa_no_eot_k2_split_views_lambda1_1gpu_test \
--attn_implementation flash_attention_3 \
--compile \
--batch_size 4 \
--global_batch_size 16 \
--num_chunks 2 \
--jepa_lambda 1.0 \
--epochs 1 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 1 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_chunk_jepa_no_eot_k2_split_views_lambda1_4gpu_full_epoch -n 4 -c 16 -m 100G -t 24:00:00 \
torchrun --standalone --nproc_per_node=4 train_chunk_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_chunk_jepa_no_eot_k2_split_views_lambda1_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--compile \
--batch_size 4 \
--global_batch_size 128 \
--num_chunks 2 \
--jepa_lambda 1.0 \
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

STP-style JEPA on the no-EOT pretraining data. This follows the `llm-jepa`
`linear=random_span` idea: run the full sequence once, sample a random patch,
represent spans with hidden-state differences, and align `before + after` to
`patch` with cosine loss.

Single-GPU short test:

```bash
python train_stp_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_stp_jepa_no_eot_patch512_768_normsqrt_lambda1_1gpu_test \
--attn_implementation flash_attention_3 \
--compile \
--batch_size 10 \
--global_batch_size 20 \
--stp_lambda 1.0 \
--min_patch_length 512 \
--max_patch_length 768 \
--length_norm sqrt \
--patch_times 1 \
--epochs 1 \
--max_steps 200 \
--lr 2e-4 \
--warmup_ratio 0.10 \
--dtype bf16 \
--num_workers 4 \
--prefetch_factor 4 \
--persistent_workers \
--log_steps 1 \
--save_every_epoch
```

```bash
./run_sm.sh -j pretrain_base_stp_jepa_no_eot_patch512_768_normsqrt_lambda1_4gpu_full_epoch -n 4 -c 16 -m 100G -t 12:00:00 \
torchrun --standalone --nproc_per_node=4 train_stp_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_stp_jepa_no_eot_patch512_768_normsqrt_lambda1_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--compile \
--batch_size 10 \
--global_batch_size 160 \
--stp_lambda 1.0 \
--min_patch_length 512 \
--max_patch_length 768 \
--length_norm sqrt \
--patch_times 1 \
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

Same STP setting without length normalization:

```bash
./run_sm.sh -j pretrain_base_stp_jepa_no_eot_patch512_768_normnone_lambda1_4gpu_full_epoch -n 4 -c 16 -m 100G -t 12:00:00 \
torchrun --standalone --nproc_per_node=4 train_stp_jepa.py \
--model_name Qwen/Qwen3-0.6B-Base \
--train_parquet data/pretrain/train_no_eot.parquet \
--output_dir experiments/qwen3_0p6b_base_stp_jepa_no_eot_patch512_768_normnone_lambda1_4gpu_full_epoch \
--attn_implementation flash_attention_3 \
--compile \
--batch_size 10 \
--global_batch_size 160 \
--stp_lambda 1.0 \
--min_patch_length 512 \
--max_patch_length 768 \
--length_norm none \
--patch_times 1 \
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
--eot_token '<|im_end|>' \
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
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3 Base MSE-JEPA linear-probe baseline using the mean of all EOT
token embeddings:

```bash
./run_sm.sh -j eval_base_jepa_mse_full_epoch_mean_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_event_jepa_4gpu_future2_mse_warmup10_full_epoch_v2/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_mse_full_epoch_mean_eot_linear_4gpu \
--pooling mean_eot \
--eot_attention none \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3 Base MSE-JEPA linear-probe baseline using the last EOT token
embedding:

```bash
./run_sm.sh -j eval_base_jepa_mse_full_epoch_last_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_event_jepa_4gpu_future2_mse_warmup10_full_epoch_v2/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_mse_full_epoch_last_eot_linear_4gpu \
--pooling last_eot \
--eot_attention none \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3 Base no-EOT AR-only linear-probe baseline. This masks packed
`<|im_end|>` event boundaries from attention and pools the last non-EOT token:

```bash
./run_sm.sh -j eval_base_ar_only_no_eot_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_ar_only_no_eot_4gpu_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_ar_only_no_eot_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3 Base no-EOT last-token MSE-JEPA linear-probe baseline. This uses
the same downstream pooling as the no-EOT AR-only baseline:

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_token_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_token_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3 Base no-EOT last-token MSE-JEPA with full-weight AR
linear-probe baseline:

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_token_ar1_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_ar1_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_token_ar1_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3 Base no-EOT last-token MSE-JEPA with half JEPA weight and
full AR weight linear-probe baseline:

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_token_jepa0p5_ar1_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa0p5_ar1_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_token_jepa0p5_ar1_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3 Base no-EOT last-token MSE-JEPA with full JEPA weight and
half AR weight linear-probe baseline:

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_token_jepa1_ar0p5_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa1_ar0p5_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_token_jepa1_ar0p5_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU event-last-token mean-pooling probes. These pool the last non-pad,
non-EOT token from each event by `event_ids`, then average those event
representations before the linear head. Use these to match the no-EOT
last-token JEPA pretraining source more closely:

```bash
./run_sm.sh -j eval_base_ar_only_no_eot_mean_event_last_token_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_ar_only_no_eot_4gpu_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_ar_only_no_eot_mean_event_last_token_linear_4gpu \
--pooling mean_event_last_token \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_token_ar1_mean_event_last_token_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_ar1_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_token_ar1_mean_event_last_token_linear_4gpu \
--pooling mean_event_last_token \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_token_jepa0p5_ar1_mean_event_last_token_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa0p5_ar1_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_token_jepa0p5_ar1_mean_event_last_token_linear_4gpu \
--pooling mean_event_last_token \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_token_jepa1_ar0p5_mean_event_last_token_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_token_jepa1_ar0p5_4gpu_future2_mse_warmup10_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_token_jepa1_ar0p5_mean_event_last_token_linear_4gpu \
--pooling mean_event_last_token \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU linear probes for the fixed-weight no-EOT last-to-future-last JEPA
checkpoints. These use `last_non_eot` pooling to match the best-performing
no-EOT JEPA evaluation setup from the previous runs:

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_to_future_last_jepa1_ar0p3_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa1_ar0p3_var0_4gpu_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_to_future_last_jepa1_ar0p3_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_to_future_last_jepa1_ar0p6_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa1_ar0p6_var0_4gpu_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_to_future_last_jepa1_ar0p6_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_to_future_last_jepa1_ar1_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa1_ar1_var0_4gpu_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_to_future_last_jepa1_ar1_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_to_future_last_jepa0p6_ar1_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa0p6_ar1_var0_4gpu_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_to_future_last_jepa0p6_ar1_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

```bash
./run_sm.sh -j eval_base_jepa_no_eot_last_to_future_last_jepa0p3_ar1_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_jepa_no_eot_last_to_future_last_jepa0p3_ar1_var0_4gpu_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_jepa_no_eot_last_to_future_last_jepa0p3_ar1_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU linear probe for the chunk-JEPA no-EOT checkpoint. This uses
`last_non_eot` pooling to match the no-EOT AR and JEPA evaluations:

```bash
./run_sm.sh -j eval_base_chunk_jepa_no_eot_k2_split_views_lambda1_last_non_eot_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir experiments/qwen3_0p6b_base_chunk_jepa_no_eot_k2_split_views_lambda1_4gpu_full_epoch/final \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/base_chunk_jepa_no_eot_k2_split_views_lambda1_last_non_eot_linear_4gpu \
--pooling last_non_eot \
--eot_attention all \
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

Four-GPU Qwen3-Embedding linear-probe baseline. Qwen3-Embedding uses a final
`<|endoftext|>` token as the sequence embedding position. The packed EHR
events use `<|im_end|>` as event boundaries, so this masks those boundary
tokens from attention and pools the appended `<|endoftext|>` representation:

```bash
./run_sm.sh -j eval_qwen3_embedding_appended_endoftext_linear_4gpu -n 4 -c 16 -m 100G -t 06:00:00 \
torchrun --standalone --nproc_per_node=4 eval_classifier.py \
--pretrained_dir Qwen/Qwen3-Embedding-0.6B \
--eval_parquet_dir data/eval \
--task icu_mortality \
--output_dir experiments/classifier/qwen3_embedding_appended_endoftext_linear_4gpu \
--pooling appended_token \
--eot_attention all \
--eot_token '<|im_end|>' \
--append_token '<|endoftext|>' \
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
--eot_token '<|im_end|>' \
--attn_implementation flash_attention_3 \
--dtype bf16 \
--batch_size 8 \
--epochs 6 \
--lr 1e-4 \
--num_workers 4
```

## Smoke tests

```bash
python -m py_compile med_jepa_common.py build_pretrain_data.py train_jepa.py train_chunk_jepa.py train_stp_jepa.py build_eval_data.py eval_classifier.py smoke_test.py
python smoke_test.py
```
