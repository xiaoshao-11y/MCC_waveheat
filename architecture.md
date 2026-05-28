# 代码架构与优化方法

## 项目概述

基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)，使用海光 DCU Z100 GPU 加速，4 卡线程并行。

---

## 文件结构

```
sdp/
├── compute_server.py          # 主计算程序 (float16 存储 + float32 计算, 4卡 GPU)
├── verify_output.py           # Python 版验证脚本
├── get_climatology.sh         # SLURM 提交脚本（自动选择空闲节点）
├── issues_and_solutions.md    # 问题与解决方案
├── architecture.md            # 本文档
├── readme.md                  # 项目进度与规划
├── ERA5/
│   ├── Climatology_ref/       # float64 参考输出 (92个 .nc 文件)
│   └── Climatology/           # 输出 (92个 .nc 文件)
├── logs/                      # SLURM 任务日志
└── verification/              # 验证结果 (由 verify_output.py 生成)
    ├── RMSE_clim.txt
    └── RMSE_P90.txt
```

---

## compute_server.py 架构

### 精度策略

采用 **float16 存储 + float32 计算**。CPU 上 data 数组用 float16（5.92 GB），GPU 上 `band_t.to(torch.float32)` 转为 float32 计算。RMSE < 0.001°C，精度满足赛题要求。

### 数据流

```
原始数据文件 (3060 个 .nc, 每年 102 天 × 30 年)
       │
       ▼
  ┌────────────────────────────────────┐
  │  并行 I/O (24 process workers)      │  ← ProcessPoolExecutor, year-grain pool.map
  │  netCDF4 直读 read_one_year() × 30 │    预建路径零 stat，无 inode 排序
  │         ↕ 后台线程重叠 ↕            │
  │  _detect_backend()  import torch   │  ← threading 后台线程，真正并行
  └────────────────────────────────────┘
       │
       ▼  data: (30, 102, 721, 1440) float16 = 5.92 GB
       │
  ┌────────────────────────┐
  │ _compute_torch_multi_gpu│  ← 4 卡线程并行
  │ 15 bands ÷ 4 GPUs       │
  │ 每 band 分 23天/批       │
  │ topk P90 + inference_mode│
  │ 线程共享 data_np 零拷贝  │
  └────────────────────────┘
       │
       ▼  threshold, climatology: (92, 721, 1440)
       │
  ┌────────────────────────┐
  │ 保存 92 个 .nc 文件     │  ← netCDF4 Dataset, float32
  │ Climatology/MMDD.nc     │
  └────────────────────────┘
```

### 关键模块

#### 1. 365 天日历

```python
build_365_calendar(year)  # 构建无 2月29日 的日历
get_date_strings_for_year(year)  # 每年 5/27 ~ 9/5 共 102 天
```

匹配参考实现，移除闰年 2 月 29 日。

#### 2. 并行 I/O（netCDF4 直读 + year-grain + 后台 Setup）

```python
read_one_year(args)  # 单个 worker 读一年文件，预建路径零 stat 调用
```

- 24 个进程（ProcessPoolExecutor），year-grain pool.map，每进程读一个年份的 ~102 个文件
- netCDF4 直读，预建 `date_to_path` 映射，**零 os.path.isfile、零 os.path.join、零 inode stat**
- `import torch` 在后台线程执行，与 I/O 真正并行重叠
- 读取速度：31-49s 读 3060 文件（取决于节点 Lustre 缓存状态）

#### 3. 多路径 GPU 检测

```
检测顺序:
 ① PyTorch ROCm/DCU → torch.version.hip
 ② PyTorch 通用 CUDA → 区分 AMD/NVIDIA GPU
 ③ HIP 原生接口 → pyhip/hip-python
 ④ CuPy ROCm 后端 → cupy
 ⑤ CPU 回退 → NumPy
```

每个路径都有明确的中文诊断输出。检测到多 GPU 时自动启用多卡并行。

#### 4. 多 GPU 线程并行

```
策略: 线程共享数据 + 线程各自绑定 GPU

15 bands 轮询分配:
  GPU 0: 4 bands, GPU 1: 4 bands
  GPU 2: 4 bands, GPU 3: 3 bands

每 band 内:
  band_t = torch.from_numpy(data_np[:, :, r0:r1, :]).to(gpu_id)
  band_t = band_t.to(torch.float32)       # f16 → f32 转换
  with torch.inference_mode():           # 关闭 autograd 节省显存
    windows = band_t.unfold(dim=1, size=11, step=1)  # 零拷贝滑动视图
    sub = windows.permute(1,0,4,2,3).reshape(23, 330, rows, 1440)
    _p90_fast_torch(sub, dim=1)          # topk P90 替代 nanquantile
    torch.nanmean(sub, dim=1)            # GPU 批量均值
```

关键优化：
- **topk P90**: 用 `torch.topk(k=topk_n, largest=True)` 只排序最大的 ~34 个元素，替代全量 `torch.nanquantile`（330 元素全排序）
- **inference_mode**: `torch.inference_mode()` 关闭 autograd，减少显存开销
- **SUB_BATCH_DAYS=23**: 每批处理 23 天，92 天分 4 批，减少循环次数
- **in-place ops**: `mul_().add_()` 原地运算，避免中间张量分配

为什么用线程而不是子进程：
- 线程共享地址空间，`data_np` 直接访问无需拷贝
- `torch.cuda.set_device()` 切换 GPU 上下文
- GPU 操作自动释放 GIL，4 线程真并行
- 零启动开销（子进程需重新 import torch + HIP 初始化 ~80s）

#### 5. CPU 回退

当无 GPU 可用时，逐天计算以控制内存。使用 `np.partition()` 替代 `np.nanpercentile()` 实现 O(n) 分位计算。

#### 6. 输出保存

- 格式：NetCDF4，float32 编码
- 变量：`Climmean`, `P90_sst`, `dayofyear`
- 维度：`[Lat, Lon]`（721×1440）

---

## 配置参数

```python
N_BANDS = 15           # 纬度分带数 (每带 ~48 行，适配 32 GB DCU)
SUB_BATCH_DAYS = 23    # 天内批次大小 (92/4=23)
N_LAT = 721            # 纬度分辨率
N_LON = 1440           # 经度分辨率
N_YEARS = 30           # 基线年数 (1991-2020)
N_OUTPUT_DAYS = 92     # 输出天数 (6/1 ~ 8/31)
WINDOW_SIZE = 11       # 滑动窗口 (5+1+5)
MAX_READ_WORKERS = 24  # I/O 并行数
```

---

## 优化历程

### 从数小时到 1.0 分钟

**起点 — `detech_waveheat.py` 逐像素 CPU 循环**

103 万次函数调用 × 166 DOY 循环 × `np.where` 搜索，推算数小时。

**Step 1 — 四维数组向量化 GPU（→ 396s）**

核心：把"逐像素处理时间序列"翻转为"对全局四维数组做张量运算"。

```
旧: (721×1440) 个一维数组 → 103 万次函数调用
新: (30年, 102天, 721, 1440) → 一次 unfold + nanquantile
```

- `torch.unfold()` 替代 `np.where` + for 循环（零拷贝滑动窗口）
- `torch.nanquantile()` 一次性对全网格所有像素同时排序
- 分带分批次：15 band × 10 天/批，控制显存 ~10 GB

**Step 2 — float32 精度（→ 244s）**

数据量 23.67 GB → 11.84 GB，PCIe 和显存带宽减半。DCU FP64:FP32 算力 1:1，加速来自内存而非计算。

**Step 3 — 4 卡线程并行（→ 166s）**

4 个线程共享 `data_np` 零拷贝，`torch.cuda.set_device()` 各自绑定 GPU。15 band 轮询分给 4 卡，Compute 从 99s → 37s。

**Step 4 — netCDF4 直读 + Setup/I/O 重叠（→ 114s）**

- netCDF4 直读替代 xarray（去元数据解析开销）
- HIP 初始化期间启动 I/O，重叠执行
- Compute 进一步优化至 29s

**Step 5 — 纯算法优化（→ 55s）**

- **topk P90**: `torch.topk(k=34)` 只排最大 ~10%，替代 330 元素全排序，Compute 29s → 19s
- **inference_mode**: 关闭 autograd 节省显存
- **SUB_BATCH_DAYS=23**: 92天分4批，减少循环和内存分配
- **in-place ops**: `mul_().add_()` 原地运算
- **去掉 inode sort**: 消除 3060 次 `os.stat()` Lustre MDS 调用，I/O 65s → 33s
- **预建路径零 stat**: 消除 `os.path.isfile()` 和 `os.path.join()` 开销
- **后台 import torch**: `import torch` 在后台线程与 I/O 真正并行，Setup 60s → 16-27s
- **PYTHONPYCACHEPREFIX**: .pyc 缓存到本地 SSD，Setup 加速显著
- **year-grain pool.map**: 30 年批量 map，减少进程调度开销
- **自动空闲节点选择**: `get_climatology.sh` 自动 `sinfo` 查 idle 节点 + `--nodelist` 指定

**Step 6 — float16 存储 + 节点优选（→ ~57s 冷启动）**

- CPU 存储从 float32（11.84 GB）→ float16（5.92 GB），pickle 序列化量减半
- GPU 计算仍用 float32，精度无损
- 节点优先级：f16r4n02 > f16* > fallback

### 最终性能（冷启动）

| 阶段 | 时间 | 占比 | 瓶颈 |
|------|------|------|------|
| Setup | 30-40s | ~60% | import torch + HIP 初始化（后台重叠） |
| I/O | 31-49s | ~65% | 读 3060 个 .nc 文件（Lustre 共享存储，节点差异） |
| Compute | 13-15s | ~23% | GPU topk P90 (4 卡并行) |
| Save | 1-2s | ~3% | 写 92 个 NetCDF |
| **Total** | **46-66s (0.8-1.1 min)** | | |

> Setup 和 I/O 后台线程重叠执行，wall time 取 max(Setup, I/O)。

> 相比 `detech_waveheat.py` 的数小时，累计加速约 **60 倍**。

---

## 性能演进

| 版本 | Compute | Total | real | 累积加速 |
|------|---------|-------|------|----------|
| 逐像素 CPU (原始) | 数小时 | 数小时 | — | 1x |
| f64 GPU 向量化 | ~200s | 396s | ~6.6min | ~20x |
| f32 单卡 | 99s | 244s | ~4.1min | ~32x |
| f32 4卡 线程并行 | 37s | 166s | ~2.8min | ~50x |
| + netCDF4直读 + 重叠 | 29s | 114s | ~2.1min | ~50x |
| + topk P90 + 去inode + 后台import | 18s | 55s | ~0.9min | ~60x |
| **+ float16 + 节点优选** | **13s** | **~57s** | **~1.0min** | **~60x** |

---

## 环境变量 (get_climatology.sh)

```bash
module load compiler/dtk/24.04        # 加载 DCU Toolkit
export ROCM_PATH=/public/software/compiler/rocm/dtk-24.04
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0,1,2,3   # 可见 GPU
export MAX_READ_WORKERS=24           # I/O 并行 worker 数
export PYTHONPYCACHEPREFIX=/tmp/pycache  # .pyc 缓存到本地 SSD
```

### 自动节点选择

```bash
bash get_climatology.sh              # 自动查找空闲节点提交
BAD_NODES=f16r4n02 bash get_climatology.sh  # 排除问题节点
```

脚本自动运行 `sinfo` 查找 idle 节点，优先选 f16r4n02（已知 I/O 最快），其次选 f16* 节点，最后降级到任意空闲节点。

---

## 验证流程

```bash
# 计算
bash get_climatology.sh

# 精度验证
sbatch clim_verification.sh
```

对比 `ERA5/Climatology/` 与 `ERA5/Climatology_ref/`（float64 参考），输出 RMSE 统计到 `verification/` 目录。
