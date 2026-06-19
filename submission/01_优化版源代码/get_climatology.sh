#!/bin/bash
# ============================================================
# 用法: bash get_climatology.sh  (不要用 sbatch 提交)
# 脚本会自动查找空闲 DCU 节点并提交作业
# ============================================================
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=04:00:00
#SBATCH -o /public/home/pan2174/sdp/logs/slurm-%j.out
#SBATCH -e /public/home/pan2174/sdp/logs/slurm-%j.err

# ---- 节点自动选择：仅在提交阶段运行（不在作业内运行） ----
if [ -z "${SLURM_JOB_ID}" ]; then
    PARTITION="kshdmcc2026"
    EXCLUDE_NODES="${BAD_NODES:-}"   # 设置 BAD_NODES 环境变量来排除节点

    echo "============================================"
    echo "  自动查找空闲 DCU 节点"
    echo "============================================"

    # 显示所有节点状态
    echo ""
    echo ">> 分区所有节点状态:"
    sinfo -p "${PARTITION}" -o "%15n %8t %10G %10m" 2>/dev/null || sinfo -p "${PARTITION}" -o "%15n %8t" 2>/dev/null
    echo ""

    # 查找空闲 DCU 节点
    IDLE_NODES=$(sinfo -p "${PARTITION}" -t idle -o "%n" --noheader 2>/dev/null)

    # 排除问题节点
    if [ -n "${EXCLUDE_NODES}" ]; then
        echo ">> 排除节点: ${EXCLUDE_NODES}"
        IDLE_NODES=$(echo "${IDLE_NODES}" | grep -v -E "$(echo ${EXCLUDE_NODES} | tr ',' '|')")
    fi

    if [ -z "${IDLE_NODES}" ]; then
        echo ">> WARNING: 无空闲节点，正常提交..."
        exec sbatch "$0" "$@"
    fi

    IDLE_COUNT=$(echo "${IDLE_NODES}" | wc -l)
    echo ">> 空闲节点 (${IDLE_COUNT}个): $(echo ${IDLE_NODES} | tr '\n' ' ')"

    # f17* nodes: fast I/O but extremely slow torch import (~60s vs 20s)
    # Node ranking (Competition, 30-rank MPI streaming):
    #   n02 15.0s >> n12 17.5s >> n13 18.1s (shared, noisy)
    # Strategy: f16r4n02 > f16r4n12 > f16* > fallback
    PREFERRED=$(echo "${IDLE_NODES}" | grep "^f16r4n02$")
    if [ -z "${PREFERRED}" ]; then
        PREFERRED=$(echo "${IDLE_NODES}" | grep "^f16r4n12$")
    fi
    if [ -z "${PREFERRED}" ]; then
        PREFERRED=$(echo "${IDLE_NODES}" | grep "^f16")
    fi
    if [ -z "${PREFERRED}" ]; then
        echo ">> WARNING: No f16 nodes idle, falling back to all idle nodes..."
        PREFERRED="${IDLE_NODES}"
    fi
    SELECTED=$(echo "${PREFERRED}" | head -1)
    echo ">> 选中节点: ${SELECTED}"
    echo ""

    exec sbatch --nodelist="${SELECTED}" "$0" "$@"
fi

# ============================================================
# 以下代码只在 Slurm 作业内运行
# ============================================================

echo ">>> 运行节点: $(hostname)"
echo ">>> 作业 ID: ${SLURM_JOB_ID}"

# Disable core dumps (prevents massive files from segfaults on bad nodes)
ulimit -c 0 2>/dev/null || true

# Clean /dev/shm from previous jobs (critical for MPI UCX shared memory)
find /dev/shm -maxdepth 1 -user "$(id -un)" -delete 2>/dev/null || true
# Fallback: tell UCX to use /tmp if /dev/shm is full
export UCX_TMP_DIR="${TMPDIR:-/tmp}"
export UCX_MEMTYPE_CACHE=n

# ---- Speed up Python imports: cache .pyc on local SSD, not Lustre ----
# Try /tmp first, fall back if full
if mkdir -p /tmp/pycache 2>/dev/null && [ -w /tmp/pycache ]; then
    export PYTHONPYCACHEPREFIX="/tmp/pycache"
else
    # /tmp full - use home dir .cache (still better than Lustre source tree)
    export PYTHONPYCACHEPREFIX="${HOME}/.cache/pycache"
    mkdir -p "${PYTHONPYCACHEPREFIX}"
    echo ">> /tmp full, pycache using ${PYTHONPYCACHEPREFIX}"
fi

# ---- Load DCU Toolkit ----
module load compiler/dtk/24.04 2>/dev/null || module load dtk 2>/dev/null || true

# ---- ROCm/HIP runtime ----
export LD_LIBRARY_PATH=/public/software/compiler/rocm/dtk-24.04/lib:/public/software/compiler/rocm/dtk-24.04/hip/lib:${LD_LIBRARY_PATH}
export ROCM_PATH=${ROCM_PATH:-/public/software/compiler/rocm/dtk-24.04}
export HIP_VISIBLE_DEVICES=0,1,2,3
MPI_NP="${MPI_NP:-30}"

# ---- PyTorch HIP alloc conf (ROCm memory tuning) ----
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-max_split_size_mb:256,garbage_collection_threshold:0.8}"

mkdir -p /public/home/pan2174/sdp/ERA5/Climatology /public/home/pan2174/sdp/logs

# ---- Locate Python directly (skip slow conda activate) ----
CONDA_ENV=/public/home/pan2174/install/miniconda3/envs/waveheat
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"

# ---- Intel MPI / ROMIO tuning for Lustre ----
export ROMIO_CB_READ=enable
export ROMIO_DS_READ=enable
export CB_BUFFER_SIZE=4194304

# Load Intel MPI module (already loaded by ls/run_climatology.sh:35)
module load mpi/hpcx/2.7.4/gcc-7.3.1 2>/dev/null || true

echo ">>> MPI mode: mpirun -np ${MPI_NP}"
{ time mpirun -np "${MPI_NP}" python /public/home/pan2174/sdp/compute_server_mpi.py; } 2>&1
