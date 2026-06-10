# 代码架构与优化方法

## 项目概述

基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)，使用海光 DCU Z200 GPU 加速。主流程为 MPI (30 ranks) + 4GPU 线程并行，float16 全程计算。

---

## 文件结构

```
sdp/
├── compute_server.py          # ProcessPool 版本（备用，24 workers）
├── compute_server_mpi.py      # MPI 版本（主流程，30 ranks + 4GPU）
├── verify_output.py           # Python 版验证脚本
├── get_climatology.sh         # SLURM 提交脚本（自动选节点，USE_MPI=1）
├── sweep_params.sh            # 参数扫描脚本
├── issues_and_solutions.md    # 问题与解决方案
├── architecture.md            # 本文档
├── readme.md                  # 项目进度与规划
├── ERA5/
│   └── Climatology/           # 输出 (92个 .nc 文件)
├── logs/                      # SLURM 任务日志
└── verification/              # 验证结果
```

---

## compute_server.py 架构

### 精度策略

**float16 全程**。存储用 float16（5.92 GB），GPU 上 float16 直接做 topk/nanmean。不转 float32，分析确认 topk 是瓶颈（占 71%），消除转换开销。RMSE < 0.001°C，满足赛题要求。

### 数据流 (MPI 版本)

```
原始数据文件 (3060 个 .nc, 30年 × 102天)
       │
       ▼
  ┌────────────────────────────────────┐
  │  Prep: 文件索引 + 格式探测           │  ← ~8s, 不计时
  │  探测 raw byte I/O offset           │
  └────────────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │  Setup: backend 检测 + import torch │  ← ~18s, 不计时
  │  PyTorch ROCm/DCU → 4 GPUs         │
  └────────────────────────────────────┘
       │
       ▼  ─── total_start (计时起点) ───
       │
  ┌────────────────────────────────────┐
  │  MPI 分布式 I/O                     │  ← 30 ranks, raw byte seek+read
  │  每 rank 独立读 1 年 ~102 文件       │
  │  float16 直读 (5.69 GB)             │    read ~5.5s
  │  direct-to-shm (零 gather)          │    各 rank 直接写共享内存
  └────────────────────────────────────┘
       │
       ▼  data: (30, 102, 721, 1440) float16 (共享内存)
       │
  ┌────────────────────────────────────┐
  │  跳过全量 copy，shm 直接读            │  ← band 级 .copy() 保证 DMA 连续
  └────────────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │  Compute: _compute_torch            │  ← 4 卡线程并行
  │  N_BANDS bands ÷ 4 GPUs            │
  │  unfold(dim=1, size=11, step=1)     │    零拷贝窗口
  │  permute(1,0,4,2,3).reshape()      │    单次连续拷贝
  │  HIP fused kernel (f16)            │    ~7.4s (NB=16)
  │    nanmean + NaN→inf + top-34 P90  │    单 kernel 完成
  │  线程共享 data_np，.copy() 确保连续  │
  └────────────────────────────────────┘
       │
       ▼  threshold, climatology: (92, 721, 1440) float32
       │
  ┌────────────────────────────────────┐
  │  Save: 92 个 .nc 文件               │  ← netCDF4 Dataset, float32
  │  Climatology/MMDD.nc                │    ~1.6s
  └────────────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │  Validation (计时后)                 │  ← nanmin/nanmax/isnan 检查
  └────────────────────────────────────┘
```

### 计时架构

```
┌──────────────────────────────────────────────┐
│ Shell { time mpirun ... }                     │
│ ┌──────────────────────────────────────────┐ │
│ │ MPI 启动 + Python import (~5s)            │ │
│ │ ┌──────────────────────────────────────┐ │ │
│ │ │ main()                                │ │ │
│ │ │   Prep   (8s)    ← 不计时            │ │ │
│ │ │   Setup (18s)   ← 不计时             │ │ │
│ │ │   ─── total_start ───                │ │ │
│ │ │   I/O    (5.8s)                      │ │ │
│ │ │   Compute(14.5s)                     │ │ │
│ │ │   Save   (1.5s)                      │ │ │
│ │ │   ─── total_elapsed ───              │ │ │
│ │ │   Total:  ~22s   (比赛计时)           │ │ │
│ │ │   Validation  ← 不计时                │ │ │
│ │ └──────────────────────────────────────┘ │ │
│ └──────────────────────────────────────────┘ │
│ Shell real: ~65s (含 import + prep + setup)  │
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
- Rank 0 做 `data.copy()` 从 shm 拷到本地 heap (~2.5s)，避免 NUMA 跨 rank 访问拖慢 GPU compute
- 预建路径字典，零 stat 调用

#### 3. 多 GPU 线程并行

```
策略: 线程共享数据 + 各绑 GPU

N_BANDS bands 轮询分配到 4 GPU
每 GPU 2-3 bands (60-90 rows/band)

每 band 内:
  band_t = torch.from_numpy(data_np[:,:,r0:r1,:].copy()).to(gpu_id)  # .copy() 确保连续
  windows = band_t.unfold(dim=1, size=11, step=1)   # 零拷贝滑动视图
  with torch.inference_mode():
    for d0 in range(0, 92, SUB_BATCH_DAYS):          # SUB_BATCH_DAYS=46, 2批
      sub = windows[:, d0:d1].permute(1,0,4,2,3).reshape(n_days, 330, rows, 1440)
      sub_thresh = _p90_fast_torch(sub, dim=1)         # float16 topk P90 (~4.0s)
      sub_clim = torch.nanmean(sub, dim=1)              # float16 均值 (~1.0s)
```

#### 4. HIP 融合内核 (fused_stats_kernel)

自研 HIP kernel，每个线程处理一个 (day, lat, lon) 像素：

```cpp
// 1. 遍历 330 个 float16 样本，寄存器维护 top-34 (插入排序)
// 2. 累加 sum + count 计算均值 (跳过 NaN)
// 3. P90 双线性插值: v0 + (v1-v0)*0.1
// 4. 64-bit 寻址防止 int32 溢出
// Grid: (n_lon/256, n_days*rows), Block: (256, 1)
// 单次内存遍历完成 nanmean + P90，消除 3 次独立 kernel launch
```

编译：`torch.utils.cpp_extension.load`，build 目录缓存在 `.fused_kernel/`。首次编译 ~2-3min（hipcc），后续秒加载。

#### 5. 输出保存

- 格式：NetCDF4，float32 编码
- 变量：`Climmean`, `P90_sst`, `dayofyear`
- 维度：`[Lat, Lon]`（721×1440）

---

## 配置参数

```python
N_BANDS = 16           # 纬度分带数 (12~16 最优)
SUB_BATCH_DAYS = 92    # 天内批次 (HIP kernel 不 OOM，全量处理)
N_LAT = 721            # 纬度分辨率
N_LON = 1440           # 经度分辨率
N_YEARS = 30           # 基线年数 (1991-2020)
N_OUTPUT_DAYS = 92     # 输出天数 (6/1 ~ 8/31, DOY 152-243)
WINDOW_SIZE = 11       # 滑动窗口 (5+1+5)
IO_THREADS = 4         # 每 rank I/O 线程数
```

---

## 优化历程

### 从数小时到 ~24s real

**Step 1-5 — GPU 向量化 + topk P90 + f16** → 55s
**Step 6-8 — f16 存储 + ProcessPool + 节点优选** → 45s
**Step 9-10 — MPI + raw byte I/O + float16 全程** → 24s
**Step 11 — Profiling**：topk 占 71%
**Step 12 — Direct-to-shm I/O** 消除 gather → I/O 从 7.5s 降到 5.8s
**Step 13 — nanmean/topk 算法验证**：
  - chunked f16 nanmean vs torch.nanmean：性能相同 (nanmean 非瓶颈)
  - torch.sort 替代 topk：20.3s，更慢，淘汰
  - 结论：torch.nanmean + topk 是最优组合
**Step 14 — SUB_BATCH_DAYS=46**：把 sub tensor 从 7.3 GiB 降到 3.7 GiB，torch.nanmean 不 OOM
**Step 15 — HIP 融合内核**：自研 HIP kernel，nanmean + NaN→inf + top-34 P90 单次遍历，Compute 14.5→7.4s
**Step 16 — 去全量 shm copy**：直接读共享内存，省 3.1s

### 最终性能（MPI, NB=16, SB=92, f16r4n03）

| 阶段 | 时间 | 说明 |
|------|------|------|
| Prep | 0.5s | 文件索引 + 格式探测 |
| Setup | 19s | backend + kernel compile（首次 2-3min） |
| I/O | 5.4s | MPI 30 ranks, direct-to-shm (zero gather) |
| Compute | 7.4s | 4×DCU Z200, HIP fused kernel f16 |
| Save | 1.5s | 92 个 NetCDF |
| **Total** | **14.5s** | I/O+Compute+Save（比赛计时） |
| **Grand** | **33.5s** | Setup + Total |
| **Shell real** | **~42s** | 含 import + prep |

### Compute per-band（HIP fused, NB=16, 45rows/band）

| 操作 | 首band | 后续band |
|------|--------|---------|
| CPU→GPU + unfold + kernel | ~3.8s | ~1.1s |
| GPU→CPU 回传 | ~0.2s | ~0.2s |

> 首 band 含 GPU JIT/冷启动预热，后续全速。每 GPU 4 bands 总 ~7.4s。

---

## 性能演进

| 版本 | Compute | I/O | Total | 加速 |
|------|---------|-----|-------|------|
| 逐像素 CPU | 数小时 | — | 数小时 | 1x |
| f32 4卡 线程并行 | 37s | 29s | — | ~50x |
| + topk P90 + f16 | 16s | 15s | ~32s | ~60x |
| + MPI Gather I/O | 14.9s | 7.5s | ~24s | ~70x |
| + Direct-to-shm + torch.nanmean | 14.5s | 5.8s | ~22s | ~75x |
| **+ HIP fused kernel + no copy** | **7.4s** | **5.4s** | **~14.5s** | **~110x** |

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

```bash
USE_MPI=1 bash get_climatology.sh
```

优先选 **f16r4n13 > f16r4n03 > f16r4n12**，其次 f16*。

---

## 验证流程

```bash
# 计算
USE_MPI=1 bash get_climatology.sh

# 精度验证
sbatch clim_verification.sh
```

对比 `ERA5/Climatology/` 与参考输出，RMSE 统计到 `verification/` 目录。
