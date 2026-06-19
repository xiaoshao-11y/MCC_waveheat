# 海温气候态计算 — DCU GPU 加速

## 项目背景

MCC 海洋计算挑战赛，基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)。算法为 11 天滑动窗口，30 年基线。

## 当前阶段

**MPI + 4GPU 并行 + HIP 边读边算流水线 + 并行 Save，Competition ~15s（比赛规则全流程计时）。**

| 阶段 | 状态 |
|------|------|
| 算法验证 | ✓ 与 float64 参考一致，RMSE < 0.001°C |
| MPI I/O | ✓ 30 ranks 分布式读, direct-to-shm 零 gather |
| HIP 流式内核 | ✓ accumulate_year + finalize, I/O 与 Compute 重叠 |
| 多 GPU 线程并行 | ✓ N_BANDS=12 ÷ 4 GPUs, 每 GPU 3 bands |
| 边读边算流水线 | ✓ ready_flags 同步, GPU 线程轮询年份就绪即处理 |
| 比赛规则适配 | ✓ torch/HIP init 排除计时，路径直接构造 |
| 并行 Save | ✓ multiprocessing fork COW, 4 进程 |

### 最终性能（N_BANDS=12, IO_THREADS=2, SAVE_WORKERS=4, WH=5, 边读边算流水线）

| 节点 | Prep | Setup | I/O | Compute | Save | Competition |
|------|------|-------|-----|---------|------|-------------|
| f16r4n02（最优） | ~1.7s | ~3.6s | ~5.8s | ~8.9s (6.0s+I/O+2.9s tail) | ~0.8s | **~15.0s** |
| f16r4n12 | ~2.2s | ~4.2s | ~6.2s | ~9.4s (6.7s+I/O+2.7s tail) | ~1.6s | ~17.5s |
| f16r4n13（共享） | ~1.6s | ~5.0s | ~6.8s | ~10.3s (7.0s+I/O+3.3s tail) | ~1.2s | ~18.1s |

```
─── 计时起点 (程序启动) ───
Prep:       1.7s  (imports + file index + format probe + MPI window)
Setup:      3.6s  (HIP kernel compile + 4 GPU micro-warmup; 首次编译 ~22s)
I/O:        5.8s  (3060 files raw byte I/O, MPI 30 ranks, direct-to-shm)
Compute:    8.9s  (4×DCU Z200, 每 GPU 3 bands 流式, 6.0s 被 I/O 覆盖)
Save:       0.8s  (92 个 NetCDF, 4 进程 fork COW 并行写)
─── 计时终点 (结果输出) ───
Competition: 15.0s  = Prep + Setup + I/O + compute_tail(2.9s) + Save
```

**核心优化：I/O 与 Compute 重叠** — GPU 线程在 I/O 进行中就开始计算，首轮 6.0s 计算完全被覆盖，仅剩第 2/3 轮 2.9s 在 I/O 结束后执行。
**关键：使用结合律** — P90 和 mean 跨年可逐步累加，无需等 30 年 I/O 全部完成。

### 比赛规则适配说明

| 排除项（不计时） | 实现 |
|-----------------|------|
| 运行环境加载 | `import torch` + `torch.cuda.init()` 在 `_t0` 之前 |
| 创建目录 | `mkdir` 在 Save 阶段，量极小 |

| 计入项（计时） | 说明 |
|---------------|------|
| 内存映射 | MPI 共享内存窗口创建 |
| 初始化 | HIP kernel JIT 编译 |
| 预热 | **已移除**（所有 warmup 方案在比赛计时下均为净亏损） |
| I/O + Compute + Save | 全流程

每 GPU 3 bands 总时间 ~5.5s。GPU 预热消除了 HIP kernel JIT 编译开销（首 band 从 ~3.5s → ~2.4s），剩余首 band 开销可能来自线程级 CUDA context 初始化。

节点优先策略：`f16r4n02 > f16r4n12 > f16r4n03 > f16*`（f16r4n13 共享节点性能波动大；f16r4n12 /dev/shm 偶尔有问题，但实测 I/O 较好）

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
| **GPU 预热 (warmup)** | Setup 阶段跑 dummy kernel 预编译 HIP kernel | 消除首 band JIT 开销，Compute 7.4→5.5s |
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

1. **HIP 融合内核不可替代**：torch.topk 在 8.1M 个独立序列上的通用排序比寄存器级 top-34 慢 3 倍以上（Compute 30s+ vs 8s）。
2. **边读边算利用结合律**：P90/top-34 和 mean 跨年可逐步累加，将 fused kernel 拆为 per-year accumulate + finalize，使 7.0s Compute 与 6.8s I/O 重叠，Competition 22-27s → 18.1s。
3. **Setup 成为新瓶颈**：5.0s Setup 中 HIP 驱动加载 GPU 二进制到指令缓存是硬件操作，无软件绕过。微 warmup 仅占 ~1.2s/GPU。
4. **流式模式减少 per-band 波动**：accumulate kernel 无需 unfold+permute+reshape，per-band 时间更均匀（~2.3-2.5s vs 1.4-5s）。
5. **比赛规则下 ~18s 接近当前架构天花板**：I/O 已被覆盖，Setup 5.0s 是硬件限制，剩余优化空间在 compute tail 和 Save。

## 性能演进

| 版本 | Compute | I/O | Competition | 备注 |
|------|---------|-----|-------------|------|
| 逐像素 CPU (原始) | 数小时 | — | — | 原始实现 |
| f32 4卡 线程并行 | 37s | 29s | — | 首次 GPU 加速 |
| + topk P90 + f16 | 16s | 15s | ~32s | 老计时 |
| + MPI Gather I/O | 14.9s | 7.5s | ~24s | |
| + Direct-to-shm + torch.nanmean | 14.5s | 5.8s | ~22s | |
| **+ HIP fused kernel + no full copy** | 7.4s | 5.4s | ~14.5s | 老计时 |
| **+ GPU warmup + parallel save** | 5.5s | 5.5-7.0s | ~12s | 老计时（不含 Prep/Setup） |
| **比赛规则适配** | 8-11s | 5.4-7.5s | **~22-27s** | Prep+Setup 计入 |
| **边读边算流水线 + 节点优选（当前）** | 8.9s | 5.8s | **~15.0s** | I/O 与 Compute 重叠 |

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

- [x] MPI I/O + float16 全程计算
- [x] Direct-to-shm I/O 消除 gather
- [x] HIP 融合内核（nanmean + P90 单 pass），Compute 14.5→7.4s
- [x] 去掉全量 shm copy（省 3.1s）
- [x] N_BANDS=12 调优，每 GPU 3 bands（60 rows/band）
- [x] 并行 Save（multiprocessing fork COW）
- [x] 参数扫描（IO_THREADS=2, SAVE_WORKERS=4）
- [x] 比赛规则适配：torch/HIP init 不计时，warmup 移除，路径直接构造
- [x] NB=8 测试（淘汰：band 太大 per-band 增幅超过轮次减少）
- [x] 无 HIP 内核回退测试（淘汰：torch.topk 慢 3 倍+）
- [x] 微 warmup / 并行 warmup / 全 warmup 测试（全部淘汰：净亏损）
- [x] **边读边算流水线**：accumulate_year + finalize 内核 + ready_flags 同步 + GPU 线程轮询，Competition 22-27s → 18.1s → **15.0s（节点优选）**
- [x] 条件 warmup：仅预热实际使用的内核路径（streaming→accumulate_fn，回退→fused_fn）
- [x] GPU buffer 预分配：year_gpu 张量复用，避免 30 年 × 3 bands 重复 CUDA 分配
- [x] batch kernel 测试（淘汰：np.stack 开销 + 延迟启动抵消批量收益）

## 下一步计划

| 优先级 | 方向 | 预估收益 | 说明 |
|--------|------|---------|------|
| P0 | **节点优选** | ~3s | Setup 5.0s 依赖于 DCU 驱动初始化速度，不同节点差异大 |
| P1 | 预编译 HIP .so | ~22s（仅冷启动） | 消除首次 JIT 编译。方案：预先在 `.fused_kernel/` 编译好 .so，运行时直接加载 |
| P2 | 减少 MPI rank 数 | ~1-2s | 30 rank 的 Barrier 开销。15 rank 各读 2 年 |
| P3 | 内核指令优化 | ~1s | accumulate_year_kernel 展开 11 天循环，减少分支 |
| - | 双服务器 | — | 已排除（Lustre MDS 瓶颈不变） |
| - | NB 调整 | — | 已排除（NB=12 最优） |

## 瓶颈分析（边读边算流水线, f16r4n02）

| 阶段 | 时间 | 占比 | 备注 |
|------|------|------|------|
| Prep | ~1.7s | 11% | imports + 路径构造 + 格式探测 + MPI window |
| Setup | ~3.6s | 24% | HIP kernel 编译 + 4 GPU micro-warmup；**当前最大瓶颈** |
| I/O | ~5.8s | — | MPI 30 ranks, raw byte I/O（与 Compute 重叠） |
| Compute tail | ~2.9s | 19% | 第 2/3 轮 band, I/O 完成后执行 |
| Save | ~0.8s | 5% | 4 进程 fork COW |

Setup 3.6s 仍是最大瓶颈：HIP 驱动将 GPU 二进制加载到每张卡的指令缓存是硬件操作，无软件绕过路径。微 warmup 选择真实 band 尺寸的 accumulate kernel 预热 4 张 GPU，每卡 ~0.8s。
I/O 5.8s 不再单独计入 Competition（因与 Compute 重叠），仅 compute tail 2.9s 暴露在关键路径上。

## 已排除的优化方向

| 方案 | 结果 |
|------|------|
| 双服务器 | I/O 不省（每台仍做 3060 × pread） |
| GPU warmup（所有变体） | 比赛规则下 Setup+Compute 净亏损 |
| NB=8 | 大 band per-band 增幅超轮次减少 |
| 去掉 HIP 融合内核 | torch.topk Compute > 30s |
| 按输出天分块 I/O | 非连续 seek 破坏 Lustre 顺序读性能 |
| batch kernel (accumulate_batch) | np.stack() 开销 + 延迟启动抵消批量收益 |
