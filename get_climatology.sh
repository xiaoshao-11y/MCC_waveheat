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
    # f16r4n02: slow I/O (~34s vs 26s)
    # Strategy: prefer f16* nodes (balanced), exclude known-bad ones
    GOOD_NODES=$(echo "${IDLE_NODES}" | grep "^f16")
    if [ -z "${GOOD_NODES}" ]; then
        echo ">> WARNING: No f16 nodes idle, falling back to all idle nodes..."
        GOOD_NODES="${IDLE_NODES}"
    fi
    SELECTED=$(echo "${GOOD_NODES}" | head -1)
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
export MAX_READ_WORKERS=24

mkdir -p /public/home/pan2174/sdp/ERA5/Climatology /public/home/pan2174/sdp/logs

# ---- Locate Python directly (skip slow conda activate) ----
CONDA_ENV=/public/home/pan2174/install/miniconda3/envs/waveheat
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"

{ time python /public/home/pan2174/sdp/compute_server.py; } 2>&1
