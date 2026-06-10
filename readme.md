# 海温气候态计算 — DCU GPU 加速

## 项目背景

MCC 海洋计算挑战赛，基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)。算法为 11 天滑动窗口，30 年基线。

## 当前阶段

**MPI + 4GPU 并行 + HIP 融合内核，Total ~14.5s。** 自研 HIP kernel（nanmean + P90 单次遍历），Compute 从 14.8s 降到 7.4s，已突破原 topk 瓶颈。

| 阶段 | 状态 |
|------|------|
| 算法验证 | ✓ 与 float64 参考一致，RMSE < 0.001°C |
| ProcessPool I/O | ✓ 24 workers, 14-29s（取决于节点 Lustre 质量） |
| MPI I/O | ✓ 30 ranks 分布式读, direct-to-shm 零 gather, ~5.5s |
| float16 GPU 计算 | ✓ unfold+permute+reshape + HIP fused kernel, 4GPU 线程并行 |
| 多 GPU 线程并行 | ✓ N_BANDS÷4 GPUs, 每 GPU 3-4 bands |
| HIP 融合内核 | ✓ nanmean + NaN→inf + top-34 P90 单 kernel 完成 |
| 参数调优 | ✓ N_BANDS=12~16, SUB_BATCH_DAYS=92 (全量，不再需要分批) |
| 去全量 copy | ✓ shm 直接读取，省 3.1s |

### 最终性能（f16r4n03，k=16 版本）

```
Prep:       0.5s  (file index + format probe)
Setup:      19s   (backend detection + kernel compile, 首次编译 2-3min)
─── 计时起点 ───
I/O:        5.4s  (3060 files raw byte I/O, MPI 30 ranks, direct-to-shm)
Compute:    7.4s  (4×DCU Z200, HIP fused kernel, float16)
Save:       1.5s  (92 个 NetCDF 输出)
─── 计时终点 ───
Total:      14.5s (I/O → save end, 纯比赛时间)
Grand:      33.5s (setup + I/O + compute + save)
Shell real: ~42s  (含 import + prep)
```

### Compute per-band（HIP fused, NB=16, 45rows/band）

| 操作 | 第一band | 后续band |
|------|---------|---------|
| CPU→GPU + unfold + kernel | ~3.8s | ~1.1s |
| （含 JIT 预热） | | |
| GPU→CPU 回传 | ~0.2s | ~0.2s |

首 band 稍慢（GPU 预热/JIT），后续全速。每 GPU 4 bands 总时间 ~7.4s。

节点优先策略：`f16r4n13 > f16r4n03 > f16*`（f16r4n12 /dev/shm 有问题）

## 加速方法

### I/O 优化（MPI 分布式 + raw byte I/O）

| 优化 | 原理 | 效果 |
|------|------|------|
| **mpi4py 30 ranks** | 每 rank 独立读 1 年 102 文件，无文件锁竞争 | 替代 ProcessPool pickle 开销 |
| **Raw byte I/O** | 直接 seek+read 固定偏移量，跳过 netCDF4 解析 | 单文件读取更快 |
| **Direct-to-shm** | 各 rank 直接写入 `MPI.Win.Allocate_shared` 数组 | 省掉 gather 步骤 (~2s) |
| **IO_THREADS=4** | 每 rank 内 ThreadPoolExecutor 并行读文件 | 减少串行等待 |
| **预建路径字典** | 启动时一次 glob 建好 `{YYYYMMDD: path}` | 零 stat 调用 |
| **float16 存储** | numpy 数组 float16（5.69 GB） | 传输量减半 |
| **node /dev/shm cleanup** | 清理前次作业残留，防止 UCX OOM | 避免 MPI 启动失败 |
| **UCX 环境变量** | `UCX_TMP_DIR`, `UCX_MEMTYPE_CACHE=n` | /dev/shm 满时降级 |

### Compute 优化（GPU 计算）

| 优化 | 原理 | 效果 |
|------|------|------|
| **HIP 融合内核** | `torch.utils.cpp_extension.load` 编译 HIP kernel，nanmean + NaN→inf + top-34 P90 单次内存遍历 | 消除 3 次独立 kernel launch，Compute 14.5→7.4s |
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
| N_BANDS | 12~16 | 纬度分带数，每 GPU 3-4 bands |
| SUB_BATCH_DAYS | 92 | 全量（HIP kernel 不 OOM，不再需要分批） |
| IO_THREADS | 4 | 每 rank I/O 线程数 |

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
| **+ HIP fused kernel + no full copy** | **7.4s** | **5.4s** | **~14.5s** | **~110x** |

## 文件结构

```
sdp/
├── compute_server.py          # ProcessPool 版本（备用）
├── compute_server_mpi.py      # MPI 版本（主流程，30 ranks）
├── verify_output.py           # 精度验证脚本
├── get_climatology.sh         # SLURM 提交脚本（自动选节点，支持 USE_MPI=1）
├── sweep_params.sh            # 参数扫描脚本（4组合×5轮）
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
- [x] N_BANDS=16 调优，首 band 后 per-band ~1.1s
- [ ] 并行 Save（92 文件多线程写入，预期 -0.5s）
- [ ] GPU 预热（消除首 band JIT 开销，预期 -1.5s）

## 下一步优化计划

### 当前状态

Total 14.5s（I/O 5.4 + Compute 7.4 + Save 1.5），Grand 33.5s（含 Setup 19s）。

### 瓶颈分析

| 阶段 | 时间 | 占比 | 优化潜力 |
|------|------|------|---------|
| I/O | 5.4s | 37% | 低（已触 Lustre 带宽上限） |
| Compute | 7.4s | 51% | 中（首 band 预热 ~2s 可消除） |
| Save | 1.5s | 10% | 低（可并行写入 -0.5s） |

### 方案优先级

| 序号 | 方案 | 描述 | 预期收益 | 风险 |
|------|------|------|---------|------|
| 1 | GPU 预热 | 在 setup 阶段跑一次 dummy kernel，消除首 band JIT/冷启动开销 | -1.5s | 低 |
| 2 | 并行 Save | ThreadPoolExecutor 并行写 92 个 nc 文件 | -0.5s | HDF5 线程安全 |
| 3 | I/O 与 compute 流水线 | 读完几年即释放 rank 开始 compute（需改架构） | -2s | 高（架构改动大） |
| 4 | N_BANDS=8 | 减少 band 数，每 GPU 2 bands，消除第 3-4 band | 待测试 | 每 band 数据更大 |

### 目标

短期目标 Total **~12s**（预热 -1.5s + 并行 Save -0.5s + 调优 -0.5s）。
