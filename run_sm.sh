#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_sm.sh [options] <command> [args...]

Options:
  -j NAME       Slurm job name. Default: first *.py command basename, or med_jepa.
  -n GPUS       Number of GPUs. Default: 0.
  -c CPUS       CPUs per task for CPU jobs, or CPUs per GPU for GPU jobs. Default: 16.
  -m MEM        Memory for CPU jobs, or memory per GPU for GPU jobs. Default: 100G.
  -t TIME       Slurm wall time. Default: 24:00:00.
  -p PARTITION  Slurm partition.
  -A ACCOUNT    Slurm account.
  -l LOG_ROOT   Log root. Default: logs/slurm.
  -h            Show this help.

Examples:
  ./run_sm.sh -j build_pretrain -c 32 -m 180G \
    python build_pretrain_data.py --meds_dir data/raw/mimic-2.2-meds/data \
    --mimic_raw_dir data/raw/mimic-iv-2.2 --output_path data/pretrain/train.parquet \
    --seq_len 2048 --num_workers 32

  ./run_sm.sh -n 4 -j train_jepa \
    torchrun --standalone --nproc_per_node=4 train_jepa.py ...
EOF
}

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR:-}" || ! -w "${TMPDIR:-}" ]]; then
export TMPDIR="/scratch/u6dk/zduan.u6dk/tmp"
# fi

n_gpus=0
job_name=""
cpus=16
mem="100G"
time_limit="24:00:00"
partition=""
account=""
log_root="logs/slurm"

while getopts ":n:j:c:m:t:p:A:l:h" opt; do
  case "$opt" in
    n) n_gpus="$OPTARG" ;;
    j) job_name="$OPTARG" ;;
    c) cpus="$OPTARG" ;;
    m) mem="$OPTARG" ;;
    t) time_limit="$OPTARG" ;;
    p) partition="$OPTARG" ;;
    A) account="$OPTARG" ;;
    l) log_root="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$job_name" ]]; then
  job_name="med_jepa"
  for arg in "$@"; do
    if [[ "$arg" == *.py ]]; then
      job_name="$(basename "$arg" .py)"
      break
    fi
  done
fi

if ! [[ "$n_gpus" =~ ^[0-9]+$ ]]; then
  echo "-n/GPUS must be a non-negative integer." >&2
  exit 1
fi

submit_date="$(date +%Y%m%d)"
submit_stamp="$(date +%Y%m%d_%H%M%S)"
log_dir="${log_root}/${submit_date}/${job_name}"
mkdir -p "$log_dir"

command_file="${log_dir}/${submit_stamp}.cmd.txt"
job_script="${log_dir}/${submit_stamp}.job.sh"
run_command="$(printf '%q ' "$@")"
{
  printf 'submit_time=%s\n' "$(date --iso-8601=seconds)"
  printf 'submit_dir=%s\n' "$(pwd)"
  printf 'job_name=%s\n' "$job_name"
  printf 'gpus=%s\n' "$n_gpus"
  printf 'cpus=%s\n' "$cpus"
  printf 'mem=%s\n' "$mem"
  printf 'time_limit=%s\n' "$time_limit"
  printf 'command='
  printf '%s' "$run_command"
  printf '\n'
} > "$command_file"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'echo "JOB_ID: ${SLURM_JOB_ID}"\n'
  printf 'echo "JOB_NAME: ${SLURM_JOB_NAME}"\n'
  printf 'echo "HOSTNAME: $(hostname)"\n'
  printf 'echo "START_TIME: $(date --iso-8601=seconds)"\n'
  printf 'echo "SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"\n'
  printf 'echo "CMD_FILE: %s"\n' "$command_file"
  printf 'echo "JOB_SCRIPT: %s"\n' "$job_script"
  printf 'echo "CMD: %s"\n' "$run_command"
  printf 'cd "${SLURM_SUBMIT_DIR}"\n'
  printf 'if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR:-}" || ! -w "${TMPDIR:-}" ]]; then\n'
  printf '  export TMPDIR="/scratch/u6dk/zduan.u6dk/tmp"\n'
  printf 'fi\n'
  printf 'mkdir -p "${TMPDIR}"\n'
  printf 'echo "TMPDIR: ${TMPDIR}"\n'
  printf 'if command -v module >/dev/null 2>&1; then\n'
  printf '  module load cuda/12.6 || true\n'
  printf 'fi\n'
  printf 'if [[ -f "%s/miniforge3/bin/conda" ]]; then\n' "$HOME"
  printf '  eval "$(%s/miniforge3/bin/conda shell.bash hook)"\n' "$HOME"
  printf '  conda activate torch\n'
  printf 'fi\n'
  printf 'set +e\n'
  printf '%s\n' "$run_command"
  printf 'status=$?\n'
  printf 'set -e\n'
  printf 'echo "END_TIME: $(date --iso-8601=seconds)"\n'
  printf 'echo "EXIT_STATUS: ${status}"\n'
  printf 'exit "${status}"\n'
} > "$job_script"
chmod +x "$job_script"

sbatch_args=(
  --job-name="$job_name"
  --nodes=1
  --ntasks=1
  --time="$time_limit"
  --output="${log_dir}/%x_%j.out"
  --error="${log_dir}/%x_%j.err"
  --export=ALL,PYTORCH_ALLOC_CONF,TMPDIR
)

if [[ "$n_gpus" -gt 0 ]]; then
  sbatch_args+=(--gres="gpu:${n_gpus}" --cpus-per-gpu="$cpus" --mem-per-gpu="$mem")
else
  sbatch_args+=(--cpus-per-task="$cpus" --mem="$mem")
fi

if [[ -n "$partition" ]]; then
  sbatch_args+=(--partition="$partition")
fi
if [[ -n "$account" ]]; then
  sbatch_args+=(--account="$account")
fi

echo "Submitting job: ${job_name}"
echo "Resources: GPUs=${n_gpus} CPUs=${cpus} MEM=${mem} TIME=${time_limit}"
echo "Logs: ${log_dir}/%x_%j.{out,err}"
echo "Command record: ${command_file}"
echo "Job script: ${job_script}"

sbatch "${sbatch_args[@]}" "$job_script"
