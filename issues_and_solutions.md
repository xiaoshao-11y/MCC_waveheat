# DCU Z100 GPU 加速实践 —— 问题与解决方案

## 环境背景

- 平台：超算互联网平台
- 服务器：CentOS 7.6 / glibc 2.17
- 加速卡：海光 DCU Z100 (Z200SM_71)，32 GB × 4
- 软件栈：DTK (DCU Toolkit) / ROCm / HIP
- Conda 环境：waveheat (Python 3.10)

---

## 问题清单

### 问题 1：GPU 检测失败，静默回退到 CPU

**现象**：任务提交后有 `--gres=dcu:4`，但日志显示 `[CPU] No accelerator detected`

**根因**：
1. Shell 脚本没有加载 DTK 环境模块，GPU 运行时库不可见
2. `_detect_backend()` 只检查 `torch.cuda.is_available()`，缺少诊断输出

**解决**：
- 在 `get_climatology.sh` 中添加 `module load compiler/dtk/24.04`，设置 ROCm/HIP 环境变量
- 重写 `_detect_backend()` 为多路径检测，添加详细中文诊断信息

---

### 问题 2：PyTorch 安装后无法导入

**现象**：`pip install torch==1.13.1+rocm5.2` 成功，但 `import torch` 报错：
```
ImportError: libamdhip64.so: version 'hip_4.2' not found
```

**根因**：上游 ROCm PyTorch wheel 链接的是 AMD 标准 `libamdhip64.so`，需要 `hip_4.2` 符号。海光 DTK 用自研 `libgalaxyhip.so` 替代了标准 HIP 库，ABI 不兼容。这不是版本号问题，是底层库完全不同。

**解决**：放弃上游 ROCm wheel，使用海光官方编译的 DAS (Deep Learning Accelerator Suite) PyTorch。

---

### 问题 3：DAS1.8 wheel 安装失败

**现象**：`torch-2.7.1+das.opt1.dtk2604-cp310-cp310-manylinux_2_28_x86_64.whl` 报错：
```
not a supported wheel on this platform
```

**根因**：DAS1.8 的 wheel 标记为 `manylinux_2_28`，需要 glibc ≥ 2.28。服务器是 CentOS 7，glibc 只有 2.17。

**解决**：从 DAS1.0 目录找到 `manylinux2014` 兼容版本：
```
torch-2.1.0+das1.0+git00661e0.abi0.dtk2404-cp310-cp310-manylinux2014_x86_64.whl
```
DTK 24.04 对应，`manylinux2014` 兼容 glibc 2.17。

---

### 问题 4：wheel 文件名导致 pip 拒绝安装

**现象**：wget 下载时用了 `-O` 简化文件名，导致 pip 报：
```
ERROR: Invalid wheel filename (wrong number of parts)
```

**根因**：pip 根据文件名解析版本号，必须保持原始 wheel 文件名。

**解决**：下载时使用原始文件名，多个 `+` 号导致 PEP 440 校验失败时，简化版本号重命名即可。

---

### 问题 5：DCU 显存不足 (OOM)

**现象**：
```
torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 39.41 GiB
GPU 0 has a total capacity of 31.98 GiB
```

**根因**：代码处理 721 个纬向行分 6 个 band，每 band ~120 行。`reshape` 操作产生中间张量：
```
92 天 × 330 样本 × 120 行 × 1440 列 × 8 bytes = 39.4 GB > 32 GB
```

**解决**：
1. 增加 `N_BANDS` 从 6 → 15，每 band 降到 48 行
2. 改为分批次处理，每次只处理 23 个输出天：
   - 每批次张量 `23 × 330 × 48 × 1440 × 4 ≈ 1.7 GB`
   - 峰值显存 ~20 GB，安全落在 32 GB 内

---

### 问题 6：MATLAB 验证脚本崩溃

**现象**：MATLAB R2018a 在海光节点上 segfault：
```
Segmentation violation ... libmwmcr.so
```

**尝试过的方案**：
- `LD_LIBRARY_PATH` 指向 conda 新版 libstdc++ → segfault
- `LD_PRELOAD` 只覆盖 libstdc++ → 同样 segfault

**根因**：MATLAB R2018a 自身的 `libmwmcr.so` 与海光 CPU 和系统环境的兼容性问题，与库路径无关。

**解决**：编写 Python 版验证脚本 `verify_output.py`，用 xarray + numpy 执行相同逻辑，无需 MATLAB。

---

### 问题 7：多 GPU 子进程并行反而更慢

**现象**：用 `multiprocessing` spawn 4 个子进程各跑一张 GPU，Compute 从 99s 涨到 115s。

**根因**：子进程隔离 CUDA 上下文的代价太高——每个子进程重新 `import torch`（~20s）、重新初始化 HIP 运行时（~30s）、Shared memory 复制 11.84 GB 数据、Queue 序列化传输结果。4 进程启动开销 ~80s，远超并行收益。

**解决**：改用线程（threading）。线程共享地址空间 `data_np` 零拷贝，`torch.cuda.set_device()` 切换 GPU 上下文，GPU 操作自动释放 GIL 实现真并行，零启动开销。Compute 99s → 37s。

---

### 问题 8：纯 float16 compute 不可用，但 float16 存储可行

**现象**：`RuntimeError: quantile() input tensor must be either float or double dtype`

**根因**：PyTorch `nanquantile` 底层排序不支持 float16 计算。

**解决**：采用 **float16 存储 + float32 计算** 策略：
- CPU data 数组用 float16（5.92 GB vs float32 的 11.84 GB），pickle 序列化量减半
- GPU 上 `band_t.to(torch.float32)` 转为 float32 后计算，精度无损
- Compute 时间 13s，RMSE < 0.001°C

---

### 问题 9：服务器 SLURM 提交了外围旧脚本

**现象**：`sbatch get_climatology.sh` 报 MATLAB 错误，但新脚本不调 MATLAB。

**根因**：服务器 home 目录和 `sdp/` 目录各有一份 `get_climatology.sh`，home 目录下 sbatch 提的是外围旧版。

**解决**：`cd sdp && sbatch get_climatology.sh` 或 `sbatch sdp/get_climatology.sh`。

---

### 问题 10：HDF5 线程不安全导致文件损坏

**现象**：用 ThreadPoolExecutor 做并行 I/O 时，netCDF4 报 HDF5 错误：`nc4_nc4f_list_add: Assertion failed`

**根因**：netCDF4 底层 HDF5 C 库不是线程安全的，多线程同时调用 `Dataset()` 会竞争内部状态导致崩溃。

**解决**：必须使用 ProcessPoolExecutor（fork），每个进程有独立 HDF5 状态天然隔离。fork 先于 GPU 初始化，COW 开销可控。线程方案不适用于 netCDF4 I/O。

---

### 问题 11：Setup/I/O 重叠效果被共享存储波动掩盖

**现象**：HIP 初始化与 I/O 并行执行，预期总时间 = max(Setup, I/O)，实际多次跑波动大。

**根因**：共享 Lustre 文件系统波动巨大，I/O 在 31s~49s 间随机变动（取决于节点和缓存状态）。当 I/O 慢于 Setup 时，Setup 被完全隐藏但 I/O 本身成为瓶颈。

**解决**：采用后台线程方案，`import torch` 在 threading.Thread 中执行，I/O 在主线程执行，两者真正并行。Setup 30-40s，I/O 31-49s，wall = max(Setup, I/O) ≈ I/O。

---

### 问题 12：Core dump 文件占用大量磁盘空间

**现象**：sdp 目录 9.2GB，但子目录加起来只有 727MB。`ls` 发现 `core.11453`、`core.7878` 等文件。

**根因**：程序 segfault 时系统自动生成 core dump，12GB 内存快照每次 4-8GB。

**解决**：
```bash
rm -f /public/home/pan2174/sdp/core.*        # 清理已有
ulimit -c 0                                     # 在 get_climatology.sh 中禁用
```

---

### 问题 13：/tmp 满导致 PYTHONPYCACHEPREFIX 失败

**现象**：`mkdir: cannot create directory '/tmp/pycache': No space left on device`

**根因**：计算节点本地 /tmp 被其他用户占满。

**解决**：`get_climatology.sh` 中先尝试 `/tmp/pycache`，失败则回退到 `~/.cache/pycache`（Lustre，但至少可用）。

---

### 问题 14：共享节点上 Lustre 负载波动导致 I/O 缓慢

**现象**：同样代码 I/O 在 31s-49s 间波动（冷启动节点间差异更大）。

**根因**：Lustre 是共享文件系统，I/O 速度依赖于当时的总负载和节点网络连接。不同节点的 Lustre 客户端缓存效率也有差异。

**解决**：
- 自动空闲节点选择：`get_climatology.sh` 运行 `sinfo` 查 idle 节点，用 `--nodelist` 指定
- 优先选 f16r4n02（已知 I/O 最快的节点，冷启动 31s）
- 代码层面已优化到极致：零 stat 调用、预建路径、year-grain pool.map、float16 存储

---

### 问题 15：inode 排序反而拖慢 I/O

**现象**：`year_files.sort(key=lambda x: os.stat(x[1]).st_ino)` 导致 I/O 从 33s 涨到 65s。

**根因**：24 个进程同时对 3060 个文件执行 `os.stat()` 获取 inode 号，Lustre MDS (Metadata Server) 被打爆。inode 排序的本意是让同 OST 文件顺序访问减少 seek，但 stat 操作的元数据开销远超 seek 节省。

**解决**：去掉 inode 排序，文件按时间序读取（同一批文件自然时序相近，OST 分布已接近最优）。I/O 65s → 33s。

---

### 问题 16：pread 直接 I/O 优化失败

**现象**：用 `os.pread()` 跳过 HDF5 直接读原始字节，预期减少 HDF5 元数据解析和 pickle 序列化开销，实际 I/O 更慢（57-70s vs 31-49s）。

**根因**：
1. HDF5 库有内部 I/O 优化（可能使用 mmap 或最优 buffer size），简单 syscall 无法超越
2. `os.pread()` 从文件偏移 23488 读取 → 内核视为随机访问 → 不做预读（readahead）
3. HDF5 Dataset 从偏移 0 顺序读取 → 触发内核预读 → 页缓存命中率高

**解决**：放弃 pread 方案，保持 netCDF4 Dataset 直读。这是该硬件 + Lustre 环境下的 I/O 最优解。

---

## 最终可工作堆栈

| 层级 | 组件 | 版本/路径 |
|------|------|-----------|
| 操作系统 | CentOS 7.6 | glibc 2.17 |
| GPU | DCU Z100 | 32 GB × 4 |
| DTK | compiler/dtk/24.04 | `/public/software/compiler/rocm/dtk-24.04` |
| HIP | 5.7 | DTK 内置 |
| PyTorch | DAS1.0 | `torch-2.1.0+das.dtk2404` |
| Python | conda waveheat | 3.10 |
| 并行方式 | threading 4 线程 | 各绑一张 GPU |
| I/O | netCDF4 直读 | ProcessPoolExecutor 24 workers, year-grain pool.map |
| 存储精度 | float16 | 5.92 GB CPU，float32 GPU 计算 |
| 验证 | verify_output.py | Python xarray |
| .pyc 缓存 | PYTHONPYCACHEPREFIX | /tmp/pycache 或 ~/.cache/pycache |

---

## 关键经验

1. **海光 DCU ≠ 标准 ROCm**：永远用海光官方 DAS wheel，不要安装上游 ROCm PyTorch
2. **CentOS 7 选 manylinux2014**：glibc 2.17 只兼容 `manylinux2014`，`manylinux_2_28` 需要 glibc 2.28+
3. **显存规划**：32 GB 显存留 22 GB 安全余量，中间张量控制在 10 GB 以下
4. **分批次计算**：SUB_BATCH_DAYS=23，92天分 4 批，减少循环和显存分配
5. **多 GPU 用线程不用子进程**：线程零拷贝 + 零启动开销，子进程 spawn 开销远超收益
6. **float16 存储 + float32 计算是这个硬件上的最优精度**：f64 太慢，纯 f16 compute 不可用，混合精度速度精度都完美
7. **I/O 用进程不用线程**：netCDF4/HDF5 线程不安全，进程天然隔离
8. **不要做 inode 排序**：Lustre 上 stat 操作开销远超 seek 节省，时间序最接近自然 OST 分布
9. **Setup 和 I/O 后台线程重叠**：`import torch` 在 threading.Thread 执行，与 I/O 真正并行
10. **PYTHONPYCACHEPREFIX 加速 import**：.pyc 缓存到本地 SSD，避免每次从 Lustre 重新编译
11. **关闭 core dump**：`ulimit -c 0`，防止 segfault 产生数 GB 的 core 文件
12. **自动选择空闲节点**：`sinfo` 查 idle 节点 + `--nodelist` 指定，优先选 I/O 快的节点
13. **不要试图绕过 HDF5 做原始 I/O**：`os.pread()` 等直接 syscall 无法匹敌 HDF5 库的 I/O 优化
14. **Lustre 冷启动性能依赖节点**：不同节点的 Lustre 客户端缓存策略不同，选对节点比优化代码更重要

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

### 最终计时拆分（冷启动）

| 阶段 | 时间 | 占比 | 瓶颈 |
|------|------|------|------|
| Setup | 30-40s | ~60% | import torch + HIP 初始化（后台重叠） |
| I/O | 31-49s | ~65% | 读 3060 个 .nc 文件（Lustre 共享存储，节点差异） |
| Compute | 13-15s | ~23% | GPU topk P90 (4 卡并行) |
| Save | 1-2s | ~3% | 写 92 个 NetCDF |
| **Total** | **46-66s (0.8-1.1 min)** | | |

> Setup 在后台线程与 I/O 重叠，wall time ≈ max(Setup, I/O) ≈ I/O。

## 并行实现

4 个 Python 线程，每个线程绑定一张 DCU 卡：

```
data (5.92 GB, CPU 内存, float16)
       │
       ├── Thread 0: torch.cuda.set_device(0) → GPU 0: 4 bands
       ├── Thread 1: torch.cuda.set_device(1) → GPU 1: 4 bands
       ├── Thread 2: torch.cuda.set_device(2) → GPU 2: 4 bands
       └── Thread 3: torch.cuda.set_device(3) → GPU 3: 3 bands
```

线程共享 CPU 内存（零拷贝），`torch.cuda.set_device()` 切换 GPU 上下文后各自独立计算，GPU 操作自动释放 GIL 实现真并行。

### 并行资源

| 资源 | 使用量 | 比赛上限 | 合规 |
|------|--------|----------|------|
| 节点数 | 1 | 2 | ✓ |
| CPU 核心 (-n) | 32 (24 I/O workers) | 64 | ✓ |
| DCU 卡 (--gres) | 4 | 单机 4 卡 | ✓ |
| 时间 | 实际 ~1.0min | 2h | ✓ |
| 独占 (--exclusive) | 是 | 要求 | ✓ |
