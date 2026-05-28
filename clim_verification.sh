#!/bin/bash
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:1
#SBATCH --time=01:00:00
#SBATCH -o /public/home/pan2174/sdp/logs/verification/slurm-%j.out
#SBATCH -e /public/home/pan2174/sdp/logs/verification/slurm-%j.err

mkdir -p /public/home/pan2174/sdp/logs/verification /public/home/pan2174/sdp/verification

echo "========================================="
echo "  验证方式 1: MATLAB"
echo "========================================="
export PATH=/public/home/pan2174/install/bin:$PATH
export LD_PRELOAD=/public/home/pan2174/install/miniconda3/lib/libstdc++.so.6
time matlab -nodisplay -nosplash -nodesktop < /public/home/pan2174/sdp/clim_verification.m 2>&1 || echo "[MATLAB 失败，以下运行 Python 验证]"

echo ""
echo "========================================="
echo "  验证方式 2: Python"
echo "========================================="
source /public/home/pan2174/install/miniconda3/bin/activate waveheat
time python /public/home/pan2174/sdp/verify_output.py
