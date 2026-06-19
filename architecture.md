# 代码架构与优化方法

## 项目概述

基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)，使用海光 DCU Z200 GPU 加速。主流程为 MPI (30 ranks) + 4GPU 线程并行，float16 全程计算。

---

## 文件结构

```
sdp/
├── compute_server_mpi.py      # MPI + 4GPU + HIP 融合内核（主流程）
├── verify_output.py           # Python 版验证脚本
├── get_climatology.sh         # SLURM 提交脚本（自动选节点）

├── issues_and_solutions.md    # 问题与解决方案
├── architecture.md            # 本文档
├── readme.md                  # 项目进度与规划
├── ERA5/
│   └── Climatology/           # 输出 (92个 .nc 文件)
└── logs/                      # SLURM 任务日志
```

---

## 主流程架构 (compute_server_mpi.py)

### 精度策略

**float16 全程**。存储用 float16（5.92 GB），GPU 上 float16 直接输入 HIP 融合 kernel（nanmean + P90 单 pass）。不转 float32，消除额外 kernel launch。RMSE < 0.001°C，满足赛题要求。

### 数据流（边读边算流水线）

```
原始数据文件 (3060 个 .nc, 30年 × 102天)
       │
       ▼
  ┌────────────────────────────────────┐
  │  Prep: 文件索引 + 格式探测           │  ← 1.7s
  │  探测 raw byte I/O offset           │
  └────────────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │  Setup: backend 检测                │  ← 3.6s
  │  + HIP 4 kernels 编译               │    accumulate_year + finalize
  │  + 4 GPU 条件 micro-warmup          │    + fused_stats + batch (备用)
  └────────────────────────────────────┘
       │
       ▼  ─── t_io (计时起点) ───
       │
  ┌──────────────────────────┬─────────────────────────────────────┐
  │  MPI 分布式 I/O (30 ranks)│  Rank 0: 4 GPU 工作线程 (并行)      │
  │                          │                                     │
  │  逐年读取:                │  Thread 0→GPU 0: 轮询 ready_flags   │
  │  Rank N 读年第 Y 年      │    年份就绪 → copy year→GPU →       │
  │    102 files raw I/O     │    accumulate_year_kernel (~67ms/年) │
  │    → data[Y,:,r0:r1,:]  │    30 年齐 → finalize_kernel         │
  │  ready_flags[Y] = 1 ────→    D2H → 写结果到 threshold/clim     │
  │                          │                                     │
  │  ~5.8s                   │  ~6.0s (被 I/O 覆盖) + 2.9s tail    │
  └──────────────────────────┴─────────────────────────────────────┘
       │                          │
       ▼  comm.Barrier()         ▼
  ┌────────────────────────────────────┐
  │  Rank 0: join GPU threads          │  ← compute tail ~2.9s
  │  非 rank 0: return                  │
  └────────────────────────────────────┘
       │
       ▼  threshold, climatology: (92, 721, 1440) float32
       │
  ┌────────────────────────────────────┐
  │  Save: 92 个 .nc 文件               │  ← netCDF4, 4 进程 fork COW
  │  Climatology/MMDD.nc                │    ~0.8s
  └────────────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │  Validation (计时后)                 │  ← nanmin/nanmax/isnan 检查
  └────────────────────────────────────┘
```

### 计时架构（边读边算流水线）

```
┌──────────────────────────────────────────────┐
│ Shell { time mpirun ... }                     │
│ ┌──────────────────────────────────────────┐ │
│ │ MPI 启动 + Python import + torch.cuda.init │ │
│ │ ┌──────────────────────────────────────┐ │ │
│ │ │ main()  _t0 = time.time()            │ │ │
│ │ │   Prep   1.7s  (imports + index +     │ │ │
│ │ │                 format probe + window)│ │ │
│ │ │   Setup  3.6s  (kernel compile +      │ │ │
│ │ │                 micro-warmup 4 GPUs)  │ │ │
│ │ │   ─── t_io ───                       │ │ │
│ │ │   ┌───────────────┐                  │ │ │
│ │ │   │ I/O   5.8s    │ ← 与 Compute 重叠 │ │ │
│ │ │   │ Compute 6.0s  │   (首轮 band)    │ │ │
│ │ │   └───────────────┘                  │ │ │
│ │ │   Barrier + join threads             │ │ │
│ │ │   Compute tail 2.9s (第2/3轮 band)    │ │ │
│ │ │   Save   0.8s                        │ │ │
│ │ │   ─── t_end ───                      │ │ │
│ │ │   Competition: 15.0s                  │ │ │
│ │ │   Validation  ← 不计时                │ │ │
│ │ └──────────────────────────────────────┘ │ │
│ └──────────────────────────────────────────┘ │
│ Shell real: ~38s (含 import + MPI 启动)      │
└──────────────────────────────────────────────┘
```

### 关键模块

#### 1. 365 天日历

```python
build_365_calendar(year)           # 构建无 2月29日 的日历
get_date_strings_for_year(year)    # 每年 5/27 ~ 9/5 共 102 天
```

#### 2. MPI 分布式 I/O + Raw Byte

```python
# 每 rank 用 raw byte I/O 独立读自己的年份
def read_one_year_raw(year, file_list, offset, n_bytes, shape, dtype):
    with open(fp, "rb") as f:
        f.seek(header_offset + n_bytes)  # 跳过 header
        raw = f.read(n_bytes)            # 直接读二进制
    arr = np.frombuffer(raw, dtype='>f8').reshape(shape).astype('float16')
```

- 30 个 MPI rank，每 rank 独立读 1 年 ~102 文件
- Raw byte I/O：直接 seek+read 固定偏移量，跳过 netCDF4 元数据解析
- **Direct-to-shm**：各 rank 直接写入 `MPI.Win.Allocate_shared` 共享内存数组，省掉 gather 步骤 (~2s)
- 不复制全量 shm→heap，band 级 `.copy()` 直接确保 GPU DMA 连续性
- 预建路径字典，零 stat 调用

#### 3. 多 GPU 流式线程并行（边读边算）

```
策略: 4 线程共享 MPI 共享内存 + 各绑 GPU + 轮询 ready_flags

N_BANDS bands 轮询分配到 4 GPU
每 GPU 3 bands (60 rows/band)

每 band 内:
  // 分配持久化 GPU 状态
  top34 = zeros(92, rows, 1440, 34) float16  (初始化为 -inf)
  partial_sum = zeros(92, rows, 1440) float32
  partial_count = zeros(92, rows, 1440) int32
  year_gpu = empty(102, rows, 1440) float16  (预分配, 复用 30 次)

  // 轮询年份就绪
  while remaining > 0:
    for yr in range(30):
      if ready_flags[yr] and not processed[yr]:
        year_gpu.copy_(data[yr, :, r0:r1, :])  // CPU→GPU
        accumulate_fn(year_gpu, top34, partial_sum, partial_count)
        processed[yr] = True; remaining--
    if remaining > 0: sleep(1ms)

  // 30 年齐: finalize
  finalize_fn(top34, partial_sum, partial_count, p90_gpu, mean_gpu)
  threshold[:, r0:r1, :] = p90_gpu.cpu()  // GPU→CPU
  climatology[:, r0:r1, :] = mean_gpu.cpu()
```

**降级路径**: 如果 HIP 流式内核编译失败 → `streaming_ok=False` → 回退到 `_compute_torch()` 原始批量 fused kernel 路径。

#### 4. HIP 流式内核

**accumulate_year_kernel** — 每年每 band 调用一次（共 30 次/band）：

```cpp
// 寄存器加载持久化状态 (top34[34] + sum + count)
// 遍历单年 11 天滑动窗口 (vs 原始 330 天)
// NaN→+inf 后插入排序维护 top-34 (纯寄存器)
// 写回持久化状态
// Grid: (n_lon/256, n_days*rows), Block: (256, 1)
// ~67ms/年/band（每线程仅 11 次迭代 vs 原始 330 次）
```

**finalize_kernel** — 每 band 调用一次：

```cpp
// 从 top34/sum/count 计算: mean = sum/count, P90 = top[33] + (top[32]-top[33])*0.1
// <1ms/band
```

**保留内核**：`fused_stats_kernel`（原始批量内核，降级回退用）和 `accumulate_batch_kernel`（未使用，测试中因 np.stack 开销淘汰）。

编译：`torch.utils.cpp_extension.load`，4 个内核编译在一起，build 目录缓存在 `.fused_kernel/`。首次编译 ~2-3min（hipcc），后续秒加载。SHA256 源码 hash 检测变化自动重编译。

#### 5. 输出保存（并行 fork COW）

- 格式：NetCDF4，float32 编码
- 变量：`Climmean`, `P90_sst`, `dayofyear`
- 维度：`[Lat, Lon]`（721×1440）
- multiprocessing fork: COW 继承内存，4 进程并行写，零数据复制

#### 6. 条件 GPU 预热

```python
# Setup 阶段, 仅预热实际使用的内核路径
if streaming_ok:
    # 流式模式: 预热 accumulate kernel (1 行, 全经度)
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            yr = torch.zeros(102, 1, N_LON, dtype=torch.float16, device=gpu_id)
            top34 = torch.full((1, 1, N_LON, 34), float('-inf'),
                               dtype=torch.float16, device=gpu_id)
            ps = torch.zeros(1, 1, N_LON, dtype=torch.float32, device=gpu_id)
            pc = torch.zeros(1, 1, N_LON, dtype=torch.int32, device=gpu_id)
            accumulate_fn(yr, top34, ps, pc)
    torch.cuda.synchronize()
elif fused_fn is not None:
    # 降级模式: 预热 fused kernel
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            d = torch.zeros(1, N_YEARS * WINDOW_SIZE, 1, N_LON,
                            dtype=torch.float16, device=gpu_id)
            p = torch.empty(1, 1, N_LON, dtype=torch.float32, device=gpu_id)
            m = torch.empty(1, 1, N_LON, dtype=torch.float32, device=gpu_id)
            fused_fn(d, p, m)
    torch.cuda.synchronize()
```

必须使用真实 band 的经度尺寸（N_LON=1440），否则 grid_x 维度不匹配导致预热无效。每 GPU ~1.2s，总计 ~5.0s Setup。消除 HIP kernel 首次启动 JIT 编译开销。

---

## 配置参数

```python
N_BANDS = 12            # 纬度分带数 (参数扫描最优，每 GPU 3 bands)
SUB_BATCH_DAYS = 92     # 天内批次 (HIP kernel 不 OOM，全量处理)
N_LAT = 721             # 纬度分辨率
N_LON = 1440            # 经度分辨率
N_YEARS = 30            # 基线年数 (1991-2020)
N_OUTPUT_DAYS = 92      # 输出天数 (6/1 ~ 8/31, DOY 152-243)
WINDOW_SIZE = 11        # 滑动窗口 (5+1+5)
IO_THREADS = 2          # 每 rank I/O 线程数 (参数扫描最优)
SAVE_WORKERS = 4        # 并行写 nc 文件的进程数
FLAGS_PAD = 64          # ready_flags 偏移 padding 防伪共享
FLAGS_COUNT = 30        # ready_flags 条目数 (一年一个)
```

---

## 环境变量 (get_climatology.sh)

```bash
module load compiler/dtk/24.04
module load mpi/hpcx/2.7.4/gcc-7.3.1     # Intel MPI
export ROCM_PATH=/public/software/compiler/rocm/dtk-24.04
export HIP_VISIBLE_DEVICES=0,1,2,3         # 可见 GPU
export USE_MPI=1                           # 启用 MPI 版本
export MPI_NP=30                           # rank 数
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256
export ROMIO_CB_READ=enable                # Lustre 优化
export ROMIO_DS_READ=enable
export UCX_TMP_DIR=/tmp                    # /dev/shm 保护
export PYTHONPYCACHEPREFIX=/tmp/pycache
```

### 自动节点选择

优先选 **f16r4n02 > f16r4n12 > f16* > f16r4n13（共享）> f17***。

```bash
USE_MPI=1 bash get_climatology.sh
```

---

## 验证流程

```bash
# 计算
USE_MPI=1 bash get_climatology.sh

# 精度验证
sbatch clim_verification.sh
```

对比 `ERA5/Climatology/` 与参考输出，RMSE 统计到 `verification/` 目录。
