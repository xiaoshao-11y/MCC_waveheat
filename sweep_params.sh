#!/bin/bash
# ============================================================
# 参数扫描: SUB_BATCH_DAYS x N_BANDS, 每组合5轮取平均
# 用法: bash sweep_params.sh
# ============================================================
#SBATCH -J sweep
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=04:00:00
#SBATCH -o /public/home/pan2174/sdp/logs/sweep-%j.out
#SBATCH -e /public/home/pan2174/sdp/logs/sweep-%j.out

# ---- 节点自动选择 ----
if [ -z "${SLURM_JOB_ID}" ]; then
    PARTITION="kshdmcc2026"
    EXCLUDE_NODES="${BAD_NODES:-}"
    echo ">> 查找空闲 DCU 节点 ..."
    IDLE_NODES=$(sinfo -p "${PARTITION}" -t idle -o "%n" --noheader 2>/dev/null)
    if [ -n "${EXCLUDE_NODES}" ]; then
        IDLE_NODES=$(echo "${IDLE_NODES}" | grep -v -E "$(echo ${EXCLUDE_NODES} | tr ',' '|')")
    fi
    if [ -z "${IDLE_NODES}" ]; then
        echo ">> 无空闲节点, 正常排队提交 ..."
        exec sbatch "$0" "$@"
    fi
    PREFERRED=$(echo "${IDLE_NODES}" | grep "^f16r4n13$")
    [ -z "${PREFERRED}" ] && PREFERRED=$(echo "${IDLE_NODES}" | grep "^f16r4n03$")
    [ -z "${PREFERRED}" ] && PREFERRED=$(echo "${IDLE_NODES}" | grep "^f16r4n12$")
    [ -z "${PREFERRED}" ] && PREFERRED=$(echo "${IDLE_NODES}" | grep "^f16")
    [ -z "${PREFERRED}" ] && PREFERRED="${IDLE_NODES}"
    SELECTED=$(echo "${PREFERRED}" | head -1)
    echo ">> 选中节点: ${SELECTED}"
    exec sbatch --nodelist="${SELECTED}" "$0" "$@"
fi

# ============================================================
# 以下只在 Slurm 作业内运行
# ============================================================
JOB_T0=${SECONDS}
export PYTHONUNBUFFERED=1
ulimit -c 0 2>/dev/null || true

# ---- pyc cache on local SSD ----
if mkdir -p /tmp/pycache 2>/dev/null && [ -w /tmp/pycache ]; then
    export PYTHONPYCACHEPREFIX="/tmp/pycache"
fi
find /dev/shm -maxdepth 1 -user "$(id -un)" \( -name 'sem.*' -o -name 'psm_*' -o -name '*.sem' \) -delete 2>/dev/null || true

# ---- DCU toolkit + ROCm/HIP ----
module load compiler/dtk/24.04 2>/dev/null || module load dtk 2>/dev/null || true
export LD_LIBRARY_PATH=/public/software/compiler/rocm/dtk-24.04/lib:/public/software/compiler/rocm/dtk-24.04/hip/lib:${LD_LIBRARY_PATH}
export ROCM_PATH=${ROCM_PATH:-/public/software/compiler/rocm/dtk-24.04}
export HIP_VISIBLE_DEVICES=0,1,2,3

# ---- Intel MPI ----
module load mpi/hpcx/2.7.4/gcc-7.3.1 2>/dev/null || true
export ROMIO_CB_READ=enable
export ROMIO_DS_READ=enable
export CB_BUFFER_SIZE=4194304
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"

# ---- waveheat python ----
CONDA_ENV="${WAVEHEAT_CONDA_ROOT:-/public/home/pan2174/install/miniconda3}/envs/waveheat"
if [ ! -x "${CONDA_ENV}/bin/python" ]; then
    echo "ERROR: waveheat python not found: ${CONDA_ENV}/bin/python"; exit 2
fi
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
echo "PYTHON=$(which python)"
python -c "import torch, netCDF4; print('torch+netCDF4 OK')" || exit 2

mkdir -p /public/home/pan2174/sdp/ERA5/Climatology /public/home/pan2174/sdp/logs

CODE_DIR=/public/home/pan2174/sdp
SWEEP_DIR="${CODE_DIR}/logs/sweep_${SLURM_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

echo "=== sweep start === node=$(hostname) job=${SLURM_JOB_ID}"

# ---- 参数网格 ----
SBD_GRID="${SBD_GRID:-23 31 46}"
NB_GRID="${NB_GRID:-24 30}"
REPEATS="${REPEATS:-5}"

echo "SBD_GRID  = ${SBD_GRID}"
echo "NB_GRID   = ${NB_GRID}"
echo "REPEATS   = ${REPEATS}"

RESULTS="${SWEEP_DIR}/results.tsv"
echo -e "combo\tSBD\tNB\trep\ttotal_s\tio_s\tcompute_s\tstatus" > "${RESULTS}"

n_combo=0
run_idx=0

for SBD in ${SBD_GRID}; do
 for NB in ${NB_GRID}; do
  n_combo=$((n_combo + 1))
  combo="SBD${SBD}_NB${NB}"
  echo ""
  echo "########## combo ${n_combo}: ${combo} ##########"

  for rep in $(seq 1 "${REPEATS}"); do
   run_idx=$((run_idx + 1))
   rlog="${SWEEP_DIR}/${combo}_r${rep}.log"

   export SUB_BATCH_DAYS="${SBD}"
   export N_BANDS="${NB}"

   echo -n "  rep ${rep}: "
   set +e
   mpirun -np 30 python "${CODE_DIR}/compute_server_mpi.py" > "${rlog}" 2>&1
   rc=$?
   set -e

   if [ "${rc}" -ne 0 ]; then
     echo "FAILED rc=${rc}"
     echo -e "${combo}\t${SBD}\t${NB}\t${rep}\tNaN\tNaN\tNaN\tFAIL_rc${rc}" >> "${RESULTS}"
     continue
   fi

   # ---- 解析时间 ----
   total=$(grep -E '^[[:space:]]*Total:' "${rlog}" | tail -1 | sed -E 's/.*Total:[[:space:]]*([0-9.]+)s.*/\1/')
   io=$(grep -E '^[[:space:]]*I/O:[[:space:]]+[0-9]' "${rlog}" | tail -1 | sed -E 's/.*I\/O:[[:space:]]*([0-9.]+)s.*/\1/')
   comp=$(grep -E '^[[:space:]]*Compute:[[:space:]]+[0-9]' "${rlog}" | tail -1 | sed -E 's/.*Compute:[[:space:]]*([0-9.]+)s.*/\1/')

   [ -z "${total}" ] && total="NaN"
   [ -z "${io}" ] && io="NaN"
   [ -z "${comp}" ] && comp="NaN"

   echo "Total=${total}s  I/O=${io}s  Compute=${comp}s"
   echo -e "${combo}\t${SBD}\t${NB}\t${rep}\t${total}\t${io}\t${comp}\tOK" >> "${RESULTS}"
  done
 done
done

echo ""
echo "============================================================"
echo "  汇总 (按平均 Compute 升序)"
echo "============================================================"
SUMMARY="${SWEEP_DIR}/summary.csv"
awk -F'\t' '
  NR==1 { next }
  $8=="OK" && $7!="NaN" {
    key=$1; sbd=$2; nb=$3; t=$7+0;
    sum_t[key]+=$5+0; cnt_t[key]+=1;
    sum_c[key]+=t; cnt_c[key]+=1;
    sum_io[key]+=$6+0;
    SBD[key]=sbd; NB[key]=nb;
  }
  END {
    print "combo,SBD,NB,runs,avg_total_s,avg_io_s,avg_compute_s" > SUMM;
    for (k in sum_c) {
      avg_t = sum_t[k]/cnt_t[k];
      avg_io = sum_io[k]/cnt_c[k];
      avg_c = sum_c[k]/cnt_c[k];
      printf "%s,%s,%s,%d,%.3f,%.3f,%.3f\n", k, SBD[k], NB[k], cnt_c[k], avg_t, avg_io, avg_c >> SUMM;
    }
  }
' SUMM="${SUMMARY}" "${RESULTS}" | sort -t',' -k7 -n | tee "${SWEEP_DIR}/leaderboard.txt"

# 格式化打印
echo ""
echo " 排名  SBD   NB   runs  avg_total   avg_io    avg_compute"
echo " ────────────────────────────────────────────────────────"
awk -F',' 'NR>1 {
  printf "  %2d   %-4s %-4s %-4s  %-8s   %-8s  %-8s\n", NR-1, $2, $3, $4, $5, $6, $7
}' "${SUMMARY}"

BEST=$(head -1 "${SWEEP_DIR}/leaderboard.txt" 2>/dev/null)
echo ""
echo "BEST: ${BEST}"
echo "results : ${RESULTS}"
echo "summary : ${SUMMARY}"
echo "TOTAL_SWEEP_WALL=$((SECONDS - JOB_T0))s  runs=${run_idx} combos=${n_combo}"
echo "=== sweep finished ==="
