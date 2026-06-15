# 海温气候态计算 — DCU GPU 加速

## 项目背景

MCC 海洋计算挑战赛，基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)。算法为 11 天滑动窗口，30 年基线。

## 当前阶段

**MPI + 4GPU 并行 + HIP 融合内核 + GPU 预热 + 并行 Save，Total ~13.0s（n03）/ ~11.5s（n13）。**

| 阶段 | 状态 |
|------|------|
| 算法验证 | ✓ 与 float64 参考一致，RMSE < 0.001°C |
| ProcessPool I/O | ✓ 24 workers, 14-29s（取决于节点 Lustre 质量） |
| MPI I/O | ✓ 30 ranks 分布式读, direct-to-shm 零 gather, ~5.5s |
| float16 GPU 计算 | ✓ unfold+permute+reshape + HIP fused kernel, 4GPU 线程并行 |
| 多 GPU 线程并行 | ✓ N_BANDS÷4 GPUs, 每 GPU 3-4 bands |
| HIP 融合内核 | ✓ nanmean + NaN→inf + top-34 P90 单 kernel 完成 |
| 参数调优 | ✓ N_BANDS=12, SUB_BATCH_DAYS=92, IO_THREADS=2, SAVE_WORKERS=4 |
| 去全量 copy | ✓ shm 直接读取，省 3.1s |
| GPU 预热 | ✓ dummy kernel 消除首 band JIT，Compute 7.4→4.7s |
| 并行 Save | ✓ multiprocessing fork COW, Save 1.5→1.0s |

### 最终性能（N_BANDS=12, IO_THREADS=2, SAVE_WORKERS=4）

| 节点 | I/O | Compute | Save | Total |
|------|-----|---------|------|-------|
| f16r4n13（最快） | ~5.5s | ~4.5s | ~1.0s | **~11.5s** |
| f16r4n03 | ~7.0s | ~4.7s | ~1.0s | **~13.0s** |

```
Prep:       0.3s  (file index + format probe)
Setup:      ~2s   (backend detection + kernel load, 首次编译 ~20s)
─── 计时起点 ───
I/O:        5.5-7.0s  (3060 files raw byte I/O, MPI 30 ranks, direct-to-shm)
Compute:    4.5-4.7s  (4×DCU Z200, HIP fused kernel, float16, GPU 预热)
Save:       1.0s  (92 个 NetCDF, 4 进程 fork COW 并行写)
─── 计时终点 ───
Total:      11.5-13.0s (I/O → save end)
Grand:      ~17s (setup + I/O + compute + save)
Shell real: ~30s (含 import + prep)
```

### Compute per-band（HIP fused, NB=12, 60rows/band, GPU 预热后）

| 操作 | 首band | 后续band |
|------|---------|---------|
| CPU→GPU + unfold + kernel | ~1.7s | ~1.3s |
| GPU→CPU 回传 | ~0.2s | ~0.2s |

每 GPU 3 bands 总时间 ~4.7s。GPU 预热消除了 HIP kernel JIT 编译开销（首 band 从 3.8s → 1.7s）。

节点优先策略：`f16r4n13 > f16r4n03 > f16*`（f16r4n12 /dev/shm 有问题）

## 加速方法

### I/O 优化（MPI 分布式 + raw byte I/O）

| 优化 | 原理 | 效果 |
|------|------|------|
| **mpi4py 30 ranks** | 每 rank 独立读 1 年 102 文件，无文件锁竞争 | 替代 ProcessPool pickle 开销 |
| **Raw byte I/O** | 直接 seek+read 固定偏移量，跳过 netCDF4 解析 | 单文件读取更快 |
| **Direct-to-shm** | 各 rank 直接写入 `MPI.Win.Allocate_shared` 数组 | 省掉 gather 步骤 (~2s) |
| **IO_THREADS=2** | 每 rank 内 ThreadPoolExecutor 并行读文件 | 参数扫描最优，减少 Lustre MDS 压力 |
| **预建路径字典** | 启动时一次 glob 建好 `{YYYYMMDD: path}` | 零 stat 调用 |
| **float16 存储** | numpy 数组 float16（5.92 GB） | 传输量减半 |
| **node /dev/shm cleanup** | 清理前次作业残留，防止 UCX OOM | 避免 MPI 启动失败 |
| **UCX 环境变量** | `UCX_TMP_DIR`, `UCX_MEMTYPE_CACHE=n` | /dev/shm 满时降级 |

### Save 优化（并行写入）

| 优化 | 原理 | 效果 |
|------|------|------|
| **multiprocessing fork** | COW 继承内存，无需复制 92×721×1440 数组 | 零启动开销 |
| **4 进程并行** | 每进程写 ~23 个文件到 NetCDF4 | Save 1.5→1.0s |
| **NETCDF4 格式** | 高效写入，不选 NETCDF3（Lustre seek 性能差） | 避免 16s 写入 |

### Compute 优化（GPU 计算）

| 优化 | 原理 | 效果 |
|------|------|------|
| **HIP 融合内核** | `torch.utils.cpp_extension.load` 编译 HIP kernel，nanmean + NaN→inf + top-34 P90 单次内存遍历 | 消除 3 次独立 kernel launch，Compute 14.5→7.4s |
| **GPU 预热 (warmup)** | Setup 阶段跑 dummy kernel 预编译 HIP kernel | 消除首 band JIT 开销，Compute 7.4→4.7s |
| **unfold + permute + reshape** | 单次连续拷贝做维度合并，替代 advanced indexing | 更快的内存布局转换 |
| **float16 全程** | 不做 f16→f32 转换，直接 f16 输入 kernel | 省显存和省转换开销 |
| **torch.inference_mode()** | 关闭 autograd 计算图追踪 | 省显存 |
| **4 卡线程并行** | 4 线程共享 data，各自绑 GPU | 零启动开销 |
| **.copy() 确保连续** | numpy 切片跨天不连续，.copy() 避免 264k 次小 DMA | GPU 传输从 row-by-row → 1次 bulk |
| **去掉全量 shm copy** | 不复制 5.92 GB 到堆内存，直接读 shm | 省 3.1s |

### 参数说明

| 参数 | 值 | 影响 |
|------|-----|------|
| WINDOW_HALF | 5 | 滑动窗口半径（固定，变影响精度） |
| N_BANDS | 12 | 纬度分带数，每 GPU 3 bands（60 rows/band） |
| SUB_BATCH_DAYS | 92 | 全量（HIP kernel 不 OOM，不再需要分批） |
| IO_THREADS | 2 | 每 rank I/O 线程数（参数扫描最优） |
| SAVE_WORKERS | 4 | 并行写 nc 文件的进程数 |

### HIP 融合内核技术细节

```cpp
// 每个线程处理一个 (day, lat, lon) 像素
// 1. 遍历 330 个样本，寄存器维护 top-34 (插入排序)
// 2. 累加 sum + count 计算均值
// 3. 双线性插值 P90: v0 + (v1-v0)*0.1
// Grid: (n_lon/256, n_days*rows), Block: (256, 1)
// 64-bit 索引防止溢出 (day*330*rows*1440 > INT32_MAX)
```

### 关键发现

Compute 从 topk 瓶颈（67%，~4s/band）被 HIP 融合内核突破。Kernel 单次遍历 330 个 float16 值完成均值和 P90，内存带宽利用率最优。

## 性能演进

| 版本 | Compute | I/O | Total | 加速 |
|------|---------|-----|-------|------|
| 逐像素 CPU (原始) | 数小时 | — | 数小时 | 1x |
| f32 4卡 线程并行 | 37s | 29s | — | ~50x |
| + topk P90 + f16 | 16s | 15s | ~32s | ~60x |
| + MPI Gather I/O | 14.9s | 7.5s | ~24s | ~70x |
| + Direct-to-shm + torch.nanmean | 14.5s | 5.8s | ~22s | ~75x |
| **+ HIP fused kernel + no full copy** | 7.4s | 5.4s | ~14.5s | ~110x |
| **+ GPU warmup + parallel save + IO_THREADS=2** | **4.7s** | **5.5-7.0s** | **~11.5-13.0s** | **~140x** |

## 文件结构

```
sdp/
├── compute_server_mpi.py      # MPI 版本（主流程，30 ranks）
├── verify_output.py           # 精度验证脚本
├── get_climatology.sh         # SLURM 提交脚本（自动选节点）

├── architecture.md            # 架构与优化详解
├── issues_and_solutions.md    # 问题与解决方案
├── readme.md                  # 本文档
├── ERA5/Climatology/          # 输出（92个 .nc 文件）
└── logs/                      # SLURM 任务日志
```

## 已完成的优化路线

- [x] MPI I/O + float16 全程计算，Total ~24s
- [x] Direct-to-shm I/O 消除 gather，I/O 从 7.5s 降到 5.8s
- [x] chunked nanmean vs torch.nanmean：性能相同，改用 torch.nanmean (代码更简洁)
- [x] torch.sort 替代 topk：慢 40%，淘汰
- [x] 不同 N_BANDS/SB 组合验证：切分不改变总 FLOPs
- [x] Profiling 确认 compute 瓶颈在 topk（67%）
- [x] HIP 融合内核（nanmean + P90 单 pass），Compute 14.5→7.4s
- [x] 去掉全量 shm copy（省 3.1s），Total 19.2→14.5s
- [x] N_BANDS=12 调优，每 GPU 3 bands
- [x] GPU 预热（dummy kernel 消除 HIP JIT 开销，Compute 7.4→4.7s）
- [x] 并行 Save（multiprocessing fork COW，Save 1.5→1.0s）
- [x] 参数扫描（IO_THREADS=2 最优，SAVE_WORKERS=4）

## 当前状态

Total ~13.0s（n03，I/O 7.0 + Compute 4.7 + Save 1.0），n13 上预期 Total ~11.5s。

### 瓶颈分析

| 阶段 | 时间 | 占比 | 备注 |
|------|------|------|------|
| I/O | 5.5-7.0s | ~55% | 主要瓶颈，受 Lustre 带宽和文件数限制 |
| Compute | 4.5-4.7s | ~35% | 已充分优化（HIP kernel + warmup） |
| Save | ~1.0s | ~8% | 已并行优化 |

### 剩余优化方向

| 序号 | 方案 | 预期收益 | 风险 |
|------|------|---------|------|
| 1 | I/O 与 compute 流水线 | -2s | 高（架构大改） |
| 2 | 减少 Lustre MDS 争用 | 节点依赖 | 不可控 |
| 3 | N_BANDS 细调 | 小 | 已验证 12 最优 |

当前优化已经接近单节点硬件极限。
