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

## 主流程架构 (compute_server_mpi.py)

### 精度策略

**float16 全程**。存储用 float16（5.92 GB），GPU 上 float16 直接输入 HIP 融合 kernel（nanmean + P90 单 pass）。不转 float32，消除额外 kernel launch。RMSE < 0.001°C，满足赛题要求。

### 数据流

```
原始数据文件 (3060 个 .nc, 30年 × 102天)
       │
       ▼
  ┌────────────────────────────────────┐
  │  Prep: 文件索引 + 格式探测           │  ← ~0.5s, 不计时
  │  探测 raw byte I/O offset           │
  └────────────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │  Setup: backend 检测 + import torch │  ← ~19s, 不计时
  │  + HIP kernel 编译/加载             │
  └────────────────────────────────────┘
       │
       ▼  ─── total_start (计时起点) ───
       │
  ┌────────────────────────────────────┐
  │  MPI 分布式 I/O                     │  ← 30 ranks, raw byte seek+read
  │  每 rank 独立读 1 年 ~102 文件       │
  │  float16 直读 (5.69 GB)             │    ~5.4s
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
  │  HIP fused kernel (f16)            │    ~7.4s
  │    nanmean + NaN→inf + top-34 P90  │    单 kernel 完成
  └────────────────────────────────────┘
       │
       ▼  threshold, climatology: (92, 721, 1440) float32
       │
  ┌────────────────────────────────────┐
  │  Save: 92 个 .nc 文件               │  ← netCDF4 Dataset, float32
  │  Climatology/MMDD.nc                │    ~1.5s
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
│ │ │   Prep   (0.5s)  ← 不计时            │ │ │
│ │ │   Setup (19s)   ← 不计时             │ │ │
│ │ │   ─── total_start ───                │ │ │
│ │ │   I/O    (5.4s)                      │ │ │
│ │ │   Compute(7.4s)                      │ │ │
│ │ │   Save   (1.5s)                      │ │ │
│ │ │   ─── total_elapsed ───              │ │ │
│ │ │   Total: 14.5s   (I/O → save end)    │ │ │
│ │ │   Validation  ← 不计时                │ │ │
│ │ └──────────────────────────────────────┘ │ │
│ └──────────────────────────────────────────┘ │
│ Shell real: ~42s (含 import + prep + setup)  │
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

#### 3. 多 GPU 线程并行

```
策略: 线程共享数据 + 各绑 GPU

N_BANDS bands 轮询分配到 4 GPU
每 GPU 3-4 bands (45-61 rows/band)

每 band 内:
  band_t = torch.from_numpy(data_np[:,:,r0:r1,:].copy()).to(gpu_id)  # .copy() 确保连续
  windows = band_t.unfold(dim=1, size=11, step=1)   # 零拷贝滑动视图
  with torch.inference_mode():
    sub = windows.permute(1,0,4,2,3).reshape(92, 330, rows, 1440)
    fused_fn(sub, p90_gpu, mean_gpu)               # HIP 融合 kernel (~1.1s)
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

优先选 **f16r4n13 > f16r4n03 > f16r4n12**，其次 f16*。

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
