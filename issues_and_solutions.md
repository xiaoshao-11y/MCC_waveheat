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

**根因**：子进程隔离 CUDA 上下文的代价太高——每个子进程重新 `import torch`（~20s）、重新初始化 HIP 运行时（~30s）、Shared memory 复制 5.92 GB 数据、Queue 序列化传输结果。4 进程启动开销 ~80s，远超并行收益。

**解决**：改用线程（threading）。线程共享地址空间 `data_np` 零拷贝，`torch.cuda.set_device()` 切换 GPU 上下文，GPU 操作自动释放 GIL 实现真并行，零启动开销。Compute 99s → 37s。

---
### 问题 10：HDF5 线程不安全导致文件损坏

**现象**：用 ThreadPoolExecutor 做并行 I/O 时，netCDF4 报 HDF5 错误：`nc4_nc4f_list_add: Assertion failed`

**根因**：netCDF4 底层 HDF5 C 库不是线程安全的，多线程同时调用 `Dataset()` 会竞争内部状态导致崩溃。

**解决**：必须使用 ProcessPoolExecutor（fork），每个进程有独立 HDF5 状态天然隔离。fork 先于 GPU 初始化，COW 开销可控。线程方案不适用于 netCDF4 I/O。

---

### 问题 11：Setup/I/O 重叠效果被共享存储波动掩盖（ProcessPool 时代）

**现象**：HIP 初始化与 I/O 并行执行，预期总时间 = max(Setup, I/O)，实际多次跑波动大。

**根因**：共享 Lustre 文件系统波动巨大，I/O 在 33s~65s 间随机变动。

**解决**：ProcessPool 模式下采用后台线程，`import torch` 在 threading.Thread 中执行，I/O 在主线程执行。MPI 模式下 Setup（含 kernel 编译）在 I/O 之前完成，编译结果缓存后后续秒加载。

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

**现象**：同样代码 I/O 在 29s-49s 间波动（冷启动节点间差异更大）。

**根因**：Lustre 是共享文件系统，I/O 速度依赖于当时的总负载和节点网络连接。

**解决**：
- 自动空闲节点选择：`get_climatology.sh` 运行 `sinfo` 查 idle 节点，用 `--nodelist` 指定
- 代码层面已优化到极致：零 stat 调用、预建路径、MPI 30 ranks 分布式读取
- `BAD_NODES=f16r4n02 bash get_climatology.sh` 排除问题节点

---

### 问题 15：inode 排序反而拖慢 I/O

**现象**：`year_files.sort(key=lambda x: os.stat(x[1]).st_ino)` 导致 I/O 从 33s 涨到 65s。

**根因**：24 个进程同时对 3060 个文件执行 `os.stat()` 获取 inode 号，Lustre MDS (Metadata Server) 被打爆。inode 排序的本意是让同 OST 文件顺序访问减少 seek，但 stat 操作的元数据开销远超 seek 节省。

**解决**：去掉 inode 排序，文件按时间序读取（同一批文件自然时序相近，OST 分布已接近最优）。I/O 65s → 33s。

---

### 问题 16：MPI UCX /dev/shm 内存不足

**现象**：`mpirun -np 30` 报错：
```
UCX ERROR Not enough memory to write total of 4292720 bytes.
Please check that /dev/shm has more available memory.
Failed to create UCP worker
```

**根因**：前一次作业残留的共享内存文件和 POSIX semaphore 占满 `/dev/shm`，UCX 无法分配进程间通信内存。

**解决**：
```bash
find /dev/shm -maxdepth 1 -user "$(id -un)" -delete 2>/dev/null
export UCX_TMP_DIR=/tmp
export UCX_MEMTYPE_CACHE=n
```

---

### 问题 17：compute 瓶颈定位 — topk 占 67%（已通过 HIP kernel 解决）

**现象**：Compute ~14.9s，调参（SBD/NB）无显著变化。

**根因**：详细 profiling 显示 topk(P90) 占 67%，nanmean 17%，GPU→CPU 5%。其余阶段 negligible。

**解决**：自研 HIP 融合内核，将 nanmean + NaN→inf + top-34 P90 合并为单 kernel，Compute 14.5→7.4s。

### 问题 18：HIP 融合内核编译 — pkg_resources 缺失

**现象**：`load_inline` 报错 `No module named 'pkg_resources'`。

**根因**：waveheat 环境中的 setuptools 82.0.1 已移除 `pkg_resources` 模块。

**解决**：降级 setuptools：
```bash
pip install "setuptools<70"
```

---

### 问题 19：HIP 融合内核编译 — ninja 缺失

**现象**：`load_inline` 报错 `Ninja is required to load C++ extensions`。

**根因**：waveheat 环境缺少 ninja 编译工具。

**解决**：
```bash
pip install ninja
```

---

### 问题 20：load_inline JIT 编译卡死在 Lustre

**现象**：`load_inline` 在 Lustre 文件系统上 hipcc 编译极慢（>3min 无响应）。

**根因**：`load_inline` 写入临时文件到 Lustre，hipcc 编译时大量元数据操作在 Lustre 上极慢。

**解决**：改用 `torch.utils.cpp_extension.load` + 持久化 build 目录：
- 源码写入 `.fused_kernel/`（持久化目录）
- ninja 自动检测变化，源码未变则跳过编译（秒加载）
- SHA256 hash 检测源码变化，自动触发重编译

---

### 问题 21：HIP kernel VMFault — int32 溢出

**现象**：kernel 编译通过，运行时 GPU VMFault crash：
```
Invalid address access: 0x2b2a5b50e000, Error code: 1.
```

**根因**：偏移量计算 `day * (n_total * rows * n_lon)` 中，day=91 时乘积 2,637,835,200 超过 `int32` 上限（2,147,483,647），地址溢出后访问非法内存。

**解决**：所有偏移量计算改用 `long long`（64-bit）。

---

### 问题 22：首 band GPU JIT/冷启动 — 已解决

**现象**：每 GPU 首 band ~3.5s，后续 band ~1.4s。首 band 慢 ~2.5x。

**根因**：GPU kernel 首次启动时触发 HIP 运行时 JIT 编译缓存和 GPU 冷启动延迟。

**解决**：Setup 阶段在每 GPU 上用真实 band 尺寸跑 dummy kernel 预热：
```python
row_sizes = sorted(set(r1 - r0 for r0, r1 in build_bands()))
for gpu_id in range(num_gpus):
    with torch.cuda.device(gpu_id):
        for rows in row_sizes:
            d = torch.zeros(1, N_YEARS * WINDOW_SIZE, rows, N_LON,
                            dtype=torch.float16, device=gpu_id)
            p = torch.empty(1, rows, N_LON,
                            dtype=torch.float32, device=gpu_id)
            m = torch.empty(1, rows, N_LON,
                            dtype=torch.float32, device=gpu_id)
            fused_fn(d, p, m)
torch.cuda.synchronize()
```
必须用真实 band 尺寸（n_rows × 1440），否则 grid_x/grid_y 不匹配导致预热无效。首 band 从 3.5s → 2.4s，Compute 7.4→5.5s（-26%）。

---

### 问题 23：并行 Save — multiprocessing fork COW

**现象**：Save 92 个 NetCDF 文件需 ~1.5s。

**根因**：单进程串行写 92 个文件。但 ThreadPoolExecutor 不可用（HDF5 线程不安全，见问题 10）。

**解决**：multiprocessing fork + COW：
```python
ctx = mp.get_context("fork")
with ctx.Pool(processes=N_SAVE_WORKERS) as pool:
    pool.map(_save_chunk, chunks)
```
- `fork` 在 GPU 初始化前完成，子进程继承父进程内存（COW），零数据复制开销
- 每子进程有独立 HDF5 状态，天然线程安全
- 4 进程并行写，Save 1.5→1.0s

---

### 问题 25：边读边算流水线 — I/O 与 Compute 重叠

**现象**：Competition 22-27s，其中 Compute 8-11s 必须等 30 个 rank 全部完成 I/O（5.4-7.5s）后才能开始，两者完全串行。

**根因**：P90/top-34 和均值对年份具备结合律（可先算 15 年的部分 top-34，再与另外 15 年合并），但原始实现是一次性将 30 年 330 天输入 fused kernel，必须等 I/O 全完成。

**解决**：实现边读边算流水线，将 Compute 与 I/O 重叠：

1. **新 HIP 内核**：将 `fused_stats_kernel`（一次处理 330 个样本）拆分为：
   - `accumulate_year_kernel`：每年调用一次，将单年 11 天窗口数据累加到持久化 top-34/sum/count 状态（寄存器驻留，每线程 11 次迭代），~67ms/年
   - `finalize_kernel`：从累积状态计算最终 P90 和 mean，<1ms/band

2. **就绪标志同步**：MPI 共享内存扩展 30 字节 `ready_flags[]` int8 数组（在 data 数组之后，加 64 字节 padding 防伪共享）。每个 rank 读完自己的年份后设置 `ready_flags[yr_idx]=1`。单节点 x86 共享物理页，写操作对所有进程立即可见。

3. **线程架构**：rank 0 在 I/O 开始前启动 4 个 GPU 工作线程。每线程轮询 `ready_flags`（1ms 间隔），年份就绪即复制到 GPU 并启动 `accumulate_year_kernel`。30 年全部就绪后调用 `finalize_kernel` 得到 P90/mean。

4. **降级安全**：如果 HIP 新内核编译失败 → `streaming_ok=False` → 自动回退到原始 `_compute_torch()` 批量计算路径。

**效果**：Competition 22-27s → 18.1s（-18%~-33%）：
- 首轮 band 的 7.0s 计算完全被 I/O（6.8s）覆盖
- 仅剩第 2/3 轮 band 的 3.3s 计算尾量在 I/O 完成后执行

### 问题 26：按天边读边算尝试 — 淘汰

**现象**：尝试将 I/O 粒度从按年（102 文件）细化到按天（1 文件），用 `fused_stats_kernel`（n_days=1, n_total=330）逐输出天计算，期望消除 compute tail。结果 Competition 32s（Barrier 同步）和 27s（per-rank 标志），均显著慢于 per-year 方案 18.1s。

**测试数据（f16r4n13）**：

| 指标 | Per-year 流式（基线） | Per-day + Barrier | Per-day + per-rank flags |
|------|---------------------|-------------------|-------------------------|
| I/O | 6.8s | 17.1s | 11.5s |
| Compute | 10.3s | 18.5s | 15.5s |
| Compute tail | 3.3s | 0.1s | 3.7s |
| **Competition** | **18.1s** | **32.0s** | **27.0s** |

**根因 1 — Barrier 放大 Lustre 尾延迟**：102 次 Barrier 强制 30 rank 每步同步，最慢 rank 的单文件 Lustre 抖动（100-300ms）拖慢全步。换 per-rank 标志数组后 I/O 从 17.1s 降到 11.5s，但仍比 per-year 的 6.8s 差。Per-year 无中间 Barrier，fast rank 可连续读完 102 文件，尾延迟被平均化。

**根因 2 — 滑动窗口导致 11 倍冗余 GPU 读取（致命）**：`fused_stats_kernel` 每输出天独立处理 330 个样本，相邻输出天的 11 天窗口有 10 天重叠：
```
输出天 0: data[:, 0:11]  → 330 样本 → kernel 读取
输出天 1: data[:, 1:12]  → 330 样本 → kernel 读取 (10/11 重叠!)
```
每个输入天被 GPU 内核读取 **11 次**（92 × 330 / 30 / 102 = 11）。对比 per-year `accumulate_year_kernel` 每个数据元素精确处理一次（寄存器中一次累加覆盖 92 个输出天），per-day 的 GPU 带宽浪费 10 倍。Compute 从 10.3s 涨到 15.5s 直接归因于此。

**结论**：对于滑动窗口聚合类计算，按最外层维度（年）做流式累加是最优的——利用结合律在寄存器中逐步累积，每个数据元素只处理一次。细化粒度的思路虽然缩短了首次结果延迟，但由于窗口重叠导致的冗余计算抵消了全部 I/O 重叠收益。**此方案永久淘汰。**

### 问题 27：并行 warmup 4 GPU 同时预热 — 淘汰

**现象**：尝试将 Setup 中的串行 4 GPU warmup 改为 4 线程并行，期望 Setup 从 5.0s 降到 ~2.0s。结果 Setup 从 5.0s 涨到 10.3s，Competition 从 18.1s 涨到 27.7s。

**根因**：HIP 驱动层的 kernel ISA 加载/指令缓存写入在多 GPU 间存在驱动级锁，4 线程并行时产生锁竞争，反而比串行更慢。每 GPU warmup ~1.2s，但并行时变成了 4 线程抢锁，每个从 1.2s 膨胀到 ~2.5s。

**结论**：HIP 多 GPU 预热必须串行。驱动层的 GPU 二进制到指令缓存的加载操作是不可并行化的硬件/驱动限制。**此方案永久淘汰。**

---

### 问题 24：参数扫描确定最优值

**现象**：不确定 IO_THREADS、SAVE_WORKERS、SUB_BATCH_DAYS 最优值。

**解决**：`sweep_params.sh` 全量扫描 75 组合 × 4 轮：
- IO_THREADS=2 最优（4/8 导致 Lustre MDS 过载）
- SAVE_WORKERS=4 最优（更多进程无帮助，Lustre MDS 瓶颈）
- SUB_BATCH_DAYS=92 全量处理（HIP kernel 不 OOM 时不需分批）

---

## 最终可工作堆栈 (MPI)

| 层级 | 组件 | 版本/路径 |
|------|------|-----------|
| 操作系统 | CentOS 7.6 | glibc 2.17 |
| GPU | DCU Z100 | 32 GB × 4 |
| DTK | compiler/dtk/24.04 | ROCm 5.7 |
| PyTorch | DAS1.0 | torch-2.1.0+das.dtk2404 |
| MPI | Intel MPI hpcx/2.7.4 | 30 ranks 分布式 I/O |
| Python | conda waveheat | 3.10 + mpi4py |
| 并行方式 | threading 4 线程 + MPI | 各绑 GPU，data 零拷贝 |
| I/O | raw byte I/O | 直接 seek+read 二进制 |
| Compute | HIP streaming kernels | accumulate_year + finalize, I/O 重叠 |
| 编译工具 | ninja + setuptools<70 | HIP kernel JIT 编译 |
| 验证 | verify_output.py | Python xarray |
| .pyc 缓存 | PYTHONPYCACHEPREFIX | /tmp/pycache |

---

## 关键经验

1. **海光 DCU ≠ 标准 ROCm**：永远用海光官方 DAS wheel，不要安装上游 ROCm PyTorch
2. **CentOS 7 选 manylinux2014**：glibc 2.17 只兼容 `manylinux2014`，`manylinux_2_28` 需要 glibc 2.28+
3. **显存规划**：32 GB 显存留 22 GB 安全余量，中间张量控制在 10 GB 以下
4. **HIP kernel 不 OOM 时全量处理**：SUB_BATCH_DAYS=92，无需分批，减少 kernel launch 次数
5. **多 GPU 用线程不用子进程**：线程零拷贝 + 零启动开销，子进程 spawn 开销远超收益
6. **float16 全程 + HIP kernel**：f16 存储减少传输带宽，HIP kernel 直接处理 f16 数据
7. **I/O 用进程不用线程**：netCDF4/HDF5 线程不安全，进程天然隔离
8. **不要做 inode 排序**：Lustre 上 stat 操作开销远超 seek 节省，时间序最接近自然 OST 分布
9. **MPI 模式下 Setup 在 I/O 之前完成**：kernel 编译（首次 2-3min）必须在 I/O 前完成，编译结果缓存到 `.fused_kernel/`，后续秒加载
10. **PYTHONPYCACHEPREFIX 加速 import**：.pyc 缓存到本地 SSD，避免每次从 Lustre 重新编译
11. **关闭 core dump**：`ulimit -c 0`，防止 segfault 产生数 GB 的 core 文件
12. **自动选择空闲节点**：`sinfo` 查 idle 节点 + `--nodelist` 指定，避开已分配节点
13. **HIP kernel 用 torch.load 而非 load_inline**：持久化 build 目录 + ninja 缓存，避免每次 JIT 编译
14. **HIP kernel 索引用 64-bit**：`day * n_total * rows * n_lon` 超 INT32_MAX，必须用 long long
15. **降级 setuptools < 70**：高版本移除 pkg_resources，load/load_inline 依赖它
16. **HIP 融合 kernel 收益巨大**：nanean + topk 合并为单次遍历，Compute 14.5→7.4s（-49%）
17. **利用结合律实现边读边算**：P90/top-34 和 mean 跨年份可结合，将单一 fused kernel 拆为 per-year accumulate + finalize，使 GPU 计算与 I/O 重叠，Competition 22-27s → 18.1s
18. **MPI 共享内存天然适合流式同步**：int8 ready_flags 数组在共享物理页上，写操作对所有 rank 立即可见，无需显式同步原语
19. **流式粒度不是越细越好**：per-day（1 文件）vs per-year（102 文件）因滑动窗口重叠导致 11 倍 GPU 冗余读取，compute 从 10.3s 涨到 15.5s。对于窗口聚合，应选最外层维度做流式——利用结合律，每元素只处理一次
20. **HIP 多 GPU 预热必须串行**：驱动层 kernel ISA 加载存在锁竞争，4 线程并行 warmup（期望 ~2s）反而膨胀到 ~10s

---

## 性能演进

| 版本 | Compute | I/O | 备注 |
|------|---------|-----|------|
| 逐像素 CPU | 数小时 | — | 原始实现 |
| f32 4卡 + ProcessPool | 37s | 29s | 首次 GPU 加速 |
| topk P90 + f16 | 16s | 15s | ~32s Total（老计时） |
| MPI + raw I/O + f16全程 | 14.9s | 7.5s | ~24s Total |
| **HIP fused kernel + no copy** | 7.4s | 5.4s | ~14.5s Total |
| **+ GPU warmup + parallel save** | 5.5s | 5.5-7.0s | ~12s Total（老计时，不含 Prep/Setup） |
| **比赛规则适配** | 8-11s | 5.4-7.5s | **~22-27s Competition** |
| **边读边算流水线（当前）** | 10.3s | 6.8s | **~18.1s Competition**（I/O 与 Compute 重叠）|

### 当前计时拆分 (边读边算流水线, NB=12, WH=5, f16r4n13)

| 阶段 | 时间 | 占比 | 说明 |
|------|------|------|------|
| Prep | ~1.6s | 9% | imports + 路径构造 + 格式探测 + MPI window |
| Setup | ~5.0s | 28% | HIP kernel 编译 + 4 GPU micro-warmup（accumulate kernel） |
| I/O | ~6.8s | — | MPI 30 ranks, raw byte I/O, direct-to-shm（与 Compute 重叠） |
| Compute | ~10.3s | — | 4×DCU Z200 流式（7.0s 在 I/O 阶段 + 3.3s tail） |
| Save | ~1.2s | 7% | 92 个 NetCDF, 4 进程 fork COW |
| **Competition** | **~18.1s** | — | = Prep + Setup + I/O + compute_tail + Save |

Competition = 1.6 + 5.0 + 6.8 + 3.3 + 1.2 ≈ 18.1s。Compute 的 7.0s 被 I/O 完全覆盖。

### 18s 时间构成详解

```
时间轴 ──────────────────────────────────────────────────────────►

  0s    1.6s       6.6s                             13.4s  14.6s  18.1s
  │ Prep │  Setup   │                               │      │      │
  │      │          │          I/O (6.8s)            │ tail │ Save  │
  │      │          │                                │(3.3s)│(1.2s)│
  │      │          ├────────────────────────────────┤      │      │
  │      │          │  Compute round 1 (7.0s)        │ r2,r3│      │
  │      │          │  ← 被 I/O 完全覆盖 →           │      │      │
  │      │          │                                │      │      │
  ▼      ▼          ▼                                ▼      ▼      ▼
 start  Prep       GPU                               I/O    全GPU  Competition
        完成       threads                         Barrier  完成   结束
                  启动                              (6.8s)
```

### Compute per-GPU per-band（流式模式, 3 bands/GPU）

| GPU 线程 | Band 1 | Band 2 | Band 3 | 完成时间 |
|----------|--------|--------|--------|---------|
| GPU 0 (bands 0-2) | ~2.3s（被 I/O 覆盖） | ~2.4s（被 I/O 覆盖） | ~2.5s（I/O 后） | ~10.3s |
| GPU 1 (bands 3-5) | ~2.3s（被 I/O 覆盖） | ~2.4s（I/O 后） | ~2.4s（I/O 后） | ~10.3s |
| GPU 2-3 | 类似 | 类似 | 类似 | ~10.3s |

每 band 处理：30 年 × accumulate_year_kernel（~67ms/年） + finalize（<1ms） + GPU↔CPU 传输。

### 首轮 vs 后续 band 为什么流式模式下差距缩小

原始 fused kernel 需要 unfold+permute+reshape 重组数据（首 band 5-6s，后续 1.4-2.5s）。流式 accumulate kernel 直接读 year slice 无需重组，per-band 时间更均匀（~2.3-2.5s），首 band 无额外 JIT 开销。

## 流式并行实现

### 总体架构

```
                    MPI 30 ranks
               ┌─────────────────────────────────────────┐
               │ Rank 0-29: 逐年读取文件                     │
               │ 读完一年 → ready_flags[year_idx] = 1     │
               │ Direct-to-shm (零 gather)                 │
               └──────────────┬──────────────────────────┘
                              │
          data (5.92 GB, rank 0, float16, 共享内存)
          ready_flags[30] int8 (64B padding 后，共享内存)
                              │
         ┌────────────────────┴────────────────────┐
         │          Rank 0: 4 GPU 工作线程          │
         │  Thread 0: GPU 0 → 轮询 ready_flags     │
         │  Thread 1: GPU 1 → 年份就绪即累加       │
         │  Thread 2: GPU 2 → 30年齐→finalize      │
         │  Thread 3: GPU 3 → D2H→写结果           │
         └─────────────────────────────────────────┘
```

### 内核设计

**accumulate_year_kernel** — 每个 (day, lat, lon) 线程处理 1 年 11 天窗口：
- 从全局内存加载持久化状态到寄存器（top34[34] + sum + count）
- 遍历 11 天滑动窗口，在寄存器中做 NaN 检查和 top-34 插入排序
- 写回持久化状态（一次全局读 + 一次全局写）
- 每线程仅 11 次迭代（vs 原始 fused kernel 的 330 次）
- ~67ms/年/band

**finalize_kernel** — 每 band 调用一次：
- 从累积 top34/sum/count 计算最终 P90 和 mean
- <1ms/band

### 同步机制

```
ready_flags: int8[30] @ MPI shared memory offset=total_bytes+64

Rank N (任意): 读完第 Y 年 → ready_flags[Y] = 1  (plain store, x86 TSO)
Rank 0 GPU 线程: while remaining > 0:
                   for yr in range(30):
                     if ready_flags[yr] && !processed[yr]:
                       复制 year 切片到 GPU → accumulate_year_kernel()
                       processed[yr] = True
                   sleep(1ms)
```

无需原子操作或内存屏障：x86 TSO 保证 plain store 对所有核可见；1ms 轮询间隔足够（rank I/O 粒度为年级别，~100-300ms/年）。

### 正确性保证：P90 和 mean 的跨年结合律

- **mean**：`(sum1+sum2)/(cnt1+cnt2)` — 直接可结合
- **P90/top-34**：从 30 年 330 个样本中取 top-34 = 从每年 11 个样本中逐步筛选 top-34 — 可逐步插入到持久化 top-34 数组
- **NaN 处理**：NaN → +inf → 落入 top-34 顶部 → 不影响 P90 计算（最终用 count 校验）

### 并行资源

| 资源 | 使用量 | 比赛上限 | 合规 |
|------|--------|----------|------|
| 节点数 | 1 | 2 | ✓ |
| CPU 核心 (-n) | 32 | 64 | ✓ |
| DCU 卡 | 4 | 单机 4 卡 | ✓ |
| MPI ranks | 30 | — | ✓ |
| 时间 | ~18.1s | 2h | ✓ |
