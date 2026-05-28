#!/bin/bash
#SBATCH -J mcc_clim
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

# Official-style single job: 1 node, 4 DCU, float16, 92 days (DOY 152-243).
# Matches get_climatology.sh resource pattern (--gres=dcu:4 per node max).

set -e
# Disable core dumps (a crashed worker can otherwise write tens of GB)
ulimit -c 0 2>/dev/null || true
JOB_T0=${SECONDS}
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/mcc_paths.sh" ]; then
  MCC_CODE_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "/public/home/pan2174/ls/mcc_paths.sh" ]; then
  MCC_CODE_DIR="/public/home/pan2174/ls"
else
  MCC_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$MCC_CODE_DIR"
source "${MCC_CODE_DIR}/mcc_paths.sh"
echo "WORKDIR=$(pwd)"

if command -v module >/dev/null 2>&1; then
  module unload anaconda3 2>/dev/null || true
  module unload dtk DTK rocm 2>/dev/null || true
  module unload devtoolset 2>/dev/null || true
  DTK_MODULE="${MCC_DTK_MODULE:-compiler/dtk/24.04.3}"
  DEVTOOLSET_MODULE="${MCC_DEVTOOLSET_MODULE:-compiler/devtoolset/7.3.1}"
  MPI_MODULE="${MCC_MPI_MODULE:-mpi/hpcx/2.7.4/gcc-7.3.1}"
  if ! module load "$DEVTOOLSET_MODULE" 2>/dev/null; then
    echo "ERROR: cannot load required devtoolset module $DEVTOOLSET_MODULE"
    exit 2
  fi
  echo "Loaded devtoolset module: $DEVTOOLSET_MODULE"
  if ! module load "$DTK_MODULE" 2>/dev/null; then
    echo "ERROR: cannot load required module $DTK_MODULE"
    exit 2
  fi
  echo "Loaded DTK module: $DTK_MODULE"
  if ! module load "$MPI_MODULE" 2>/dev/null; then
    echo "ERROR: cannot load required MPI module $MPI_MODULE"
    exit 2
  fi
  echo "Loaded MPI module: $MPI_MODULE"
fi

export MCC_FAST_P90=1
export OMP_NUM_THREADS=1
export MCC_COMPUTE_DTYPE="${MCC_COMPUTE_DTYPE:-float16}"
export MCC_NUM_GPU="${MCC_NUM_GPU:-4}"
export MCC_GPU_WARMUP="${MCC_GPU_WARMUP:-0}"
export MCC_SUB_BATCH_DAYS="${MCC_SUB_BATCH_DAYS:-2}"
export MCC_GPU_BAND_ROWS="${MCC_GPU_BAND_ROWS:-120}"
export MCC_WORKER_PREFETCH="${MCC_WORKER_PREFETCH:-0}"
export MCC_WORKER_WRITE_THREADS="${MCC_WORKER_WRITE_THREADS:-1}"
export MCC_PRE_IMPORT_TORCH="${MCC_PRE_IMPORT_TORCH:-1}"
export MCC_IO_GRAIN="${MCC_IO_GRAIN:-year}"
export MCC_IO_WORKERS="${MCC_IO_WORKERS:-16}"
export MCC_IO_BACKEND="${MCC_IO_BACKEND:-process}"
export MCC_PREFETCH=1
export MCC_CACHE_ALL_NEEDED=1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-max_split_size_mb:256,garbage_collection_threshold:0.8}"
# Slurm may set HIP_VISIBLE_DEVICES=0 → only 1 DCU visible; clear for 4-card job
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

source "${MCC_HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mcc-gpu 2>/dev/null || conda activate mcc
echo "PYTHON=$(which python)"

echo "SLURM_GPUS=${SLURM_GPUS:-unset} MCC_NUM_GPU=${MCC_NUM_GPU} dtype=${MCC_COMPUTE_DTYPE}"
SETUP_SEC=$((SECONDS - JOB_T0))
echo "shell_setup: ${SETUP_SEC}s"

PY_T0=${SECONDS}
time python run_climatology.py \
  --nc-path "$MCC_NC_PATH" \
  --save-path "$MCC_SAVE_PATH" \
  --device auto \
  --compute-dtype "$MCC_COMPUTE_DTYPE" \
  --start-doy 152 --end-doy 243 \
  --batch-rows 721 \
  --num-gpu "$MCC_NUM_GPU" \
  --gpu-band-rows "$MCC_GPU_BAND_ROWS" \
  --io-workers "$MCC_IO_WORKERS" \
  --io-backend "$MCC_IO_BACKEND" \
  --io-grain "$MCC_IO_GRAIN" \
  --cache-all-needed 1 \
  --prefetch 1 \
  --worker-prefetch "$MCC_WORKER_PREFETCH" \
  --worker-write-threads "$MCC_WORKER_WRITE_THREADS"

PY_SEC=$((SECONDS - PY_T0))
TOTAL_SEC=$((SECONDS - JOB_T0))
echo "python_wall: ${PY_SEC}s"
echo "task_total_wall: ${TOTAL_SEC}s"
echo "=== finished ==="
