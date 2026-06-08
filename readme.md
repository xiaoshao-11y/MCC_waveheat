# 海温气候态计算 — DCU GPU 加速

## 项目背景

MCC 海洋计算挑战赛，基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)。算法为 11 天滑动窗口，30 年基线。

## 当前阶段

**MPI + 4GPU 并行，比赛计时 ~22s。** compute 瓶颈已到硬件极限（topk 占 67%），单节点 88 天全量计算的天花板。

| 阶段 | 状态 |
|------|------|
| 算法验证 | ✓ 与 float64 参考一致，RMSE < 0.001°C |
| ProcessPool I/O | ✓ 24 workers, 14-29s（取决于节点 Lustre 质量） |
| MPI I/O | ✓ 30 ranks 分布式读, direct-to-shm 零 gather, ~5.8s |
| float16 GPU 计算 | ✓ unfold+permute+reshape, topk P90, 4GPU 线程并行 |
| 多 GPU 线程并行 | ✓ N_BANDS÷4 GPUs, 每 GPU 2-3 bands |
| 参数调优 | ✓ N_BANDS=12, SUB_BATCH_DAYS=46 |
| 算法验证 | ✓ chunked nanmean/torch.nanmean/sort vs topk 全验证 |
| 详细 profiling | ✓ topk 占 67%，确认无法进一步优化 |

### 最终性能（冷启动，最佳节点 f16r4n13）

```
Prep:       9s    (file index + format probe, 不计时)
Setup:      18s   (import torch + HIP 初始化, 不计时)
─── 比赛计时起点 ───
I/O:        5.8s  (3060 files raw byte I/O, MPI 30 ranks, direct-to-shm)
Compute:   14.5s  (4×DCU Z200 并行, topk P90, float16)
Save:       1.5s  (92 个 NetCDF 输出)
─── 比赛计时终点 ───
Total:      ~22s  (I/O+Compute+Save, 比赛计时)
Shell real: ~60s  (完整进程, 含 import + prep + setup)
```

### 实测节点排名

| 节点 | I/O | Compute | Total | Shell real |
|------|-----|---------|-------|------------|
| f16r4n13 | 5.8s | 14.5s | **~22s** | ~60s |

节点优先策略：`f16r4n13 > f16r4n03 > f16r4n12 > f16*`

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
| **unfold + permute + reshape** | 单次连续拷贝做维度合并，替代 advanced indexing | 更快的内存布局转换 |
| **float16 全程** | 不做 f16→f32 转换，直接 f16 topk/nanmean | 省显存和省转换开销 |
| **topk P90** | `torch.topk(k=34)` 只排最大 ~10% | 已验证比 sort 快 40% |
| **torch.inference_mode()** | 关闭 autograd 计算图追踪 | 省显存 |
| **SUB_BATCH_DAYS=46** | 92 天分 2 批，sub tensor 从 7.3→3.7 GiB | torch.nanmean 不 OOM |
| **4 卡线程并行** | 4 线程共享 data，各自绑 GPU | 零启动开销 |
| **.copy() 确保连续** | numpy 切片跨天不连续，.copy() 避免 264k 次小 DMA | GPU 传输从 row-by-row → 1次 bulk |

### 参数说明

| 参数 | 值 | 影响 |
|------|-----|------|
| WINDOW_HALF | 5 | 滑动窗口半径（固定，变影响精度） |
| N_BANDS | 24 (默认) / 12 (推荐) | 纬度分带数，每 GPU 2-6 bands |
| SUB_BATCH_DAYS | 46 | 天数子批次，torch.nanmean 显存控制 |
| IO_THREADS | 4 | 每 rank I/O 线程数 |

### 关键发现（profiling 结果）

每个 band（60 rows, 2 sub-batches, NB=12）耗时分解：

| 阶段 | 时间 | 占比 |
|------|------|------|
| CPU→GPU 传输 | ~0.3s | 5% |
| unfold + permute + reshape | ~0.5s | 8% |
| **topk(P90)** | **~4.0s** | **67%** |
| torch.nanmean(clim) | ~1.0s | 17% |
| GPU→CPU 回传 | ~0.2s | 3% |

**结论**: compute 瓶颈是 topk（AMD DCU kernel 硬件极限）。已验证 sort 更慢（20.3s），切分更多 band 不减少总 FLOPs，torch.compile 对 topk 无帮助。

## 性能演进

| 版本 | Compute | I/O | Total | 加速 |
|------|---------|-----|-------|------|
| 逐像素 CPU (原始) | 数小时 | — | 数小时 | 1x |
| f32 4卡 线程并行 | 37s | 29s | — | ~50x |
| + topk P90 + f16 | 16s | 15s | ~32s | ~60x |
| + MPI Gather I/O | 14.9s | 7.5s | ~24s | ~70x |
| **+ Direct-to-shm + torch.nanmean** | **14.5s** | **5.8s** | **~22s** | **~75x** |

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
- [x] Profiling 确认 compute 瓶颈在 topk（67%），单节点已到硬件极限
- [ ] pin_memory + torch.compile（预期收益 ~1s，未测试）
