# 海温气候态计算 — DCU GPU 加速

## 项目背景

MCC 海洋计算挑战赛，基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)。算法为 11 天滑动窗口，30 年基线。

## 当前阶段

**MPI + 4GPU 并行，比赛计时 ~24s。** compute 瓶颈已到硬件极限（topk 占 60%），下一步做双节点各跑半个赛季。

| 阶段 | 状态 |
|------|------|
| 算法验证 | ✓ 与 float64 参考一致，RMSE < 0.001°C |
| ProcessPool I/O | ✓ 24 workers, 14-29s（取决于节点 Lustre 质量） |
| MPI I/O | ✓ 30 ranks 分布式读 + 零拷贝 gather, 5.5s 读 + 2s gather |
| float16 GPU 计算 | ✓ unfold+permute+reshape, topk P90, 4GPU 线程并行 |
| 多 GPU 线程并行 | ✓ 24 bands ÷ 4 GPU，每 GPU 6 bands |
| 参数调优 | ✓ N_BANDS=24, SUB_BATCH_DAYS=23 |
| 详细 profiling | ✓ topk 占 71%，确认无隐藏瓶颈 |
| 双节点 → 各半赛季 | 下一步 |

### 最终性能（冷启动，最佳节点 f16r4n13）

```
Prep:       9s    (file index + format probe, 不计时)
Setup:      18s   (import torch + HIP 初始化, 不计时)
─── 比赛计时起点 ───
I/O:        7.5s  (3060 files raw byte I/O, MPI 30 ranks)
Compute:    14.9s (4×DCU Z200 并行, topk P90, float16)
Save:        1.6s (92 个 NetCDF 输出)
─── 比赛计时终点 ───
Total:      ~24s  (I/O+Compute+Save, 比赛计时)
Shell real: ~65s  (完整进程, 含 import + prep + setup)
```

### 实测节点排名

| 节点 | I/O | Compute | Total | Shell real |
|------|-----|---------|-------|------------|
| f16r4n13 | 7.5s | 14.9s | **~24s** | ~65s |

节点优先策略：`f16r4n13 > f16r4n03 > f16r4n12 > f16*`

## 加速方法

### I/O 优化（MPI 分布式 + raw byte I/O）

| 优化 | 原理 | 效果 |
|------|------|------|
| **mpi4py 30 ranks** | 每 rank 独立读 1 年 102 文件，无文件锁竞争 | 替代 ProcessPool pickle 开销 |
| **Raw byte I/O** | 直接 seek+read 固定偏移量，跳过 netCDF4 解析 | 单文件读取更快 |
| **MPI_Gatherv** | 30 ranks 结果汇合到 rank 0，共享内存零拷贝 | ~2s（远快于 pickle） |
| **预建路径字典** | 启动时一次 glob 建好 `{YYYYMMDD: path}` | 零 stat 调用 |
| **float16 存储** | numpy 数组 float16（5.92 GB） | 传输量减半 |
| **node /dev/shm cleanup** | 清理前次作业残留，防止 UCX OOM | 避免 MPI 启动失败 |
| **UCX 环境变量** | `UCX_TMP_DIR`, `UCX_MEMTYPE_CACHE=n` | /dev/shm 满时降级 |

### Compute 优化（GPU 计算）

| 优化 | 原理 | 效果 |
|------|------|------|
| **unfold + permute + reshape** | 单次连续拷贝做维度合并，替代 advanced indexing | 更快的内存布局转换 |
| **float16 全程** | 不做 f16→f32 转换，直接 f16 topk/nanmean | 显存和省转换开销 |
| **topk P90** | `torch.topk(k=34)` 只排最大 ~10% | f16 下 ~370ms/次 |
| **torch.inference_mode()** | 关闭 autograd 计算图追踪 | 省显存 |
| **N_BANDS=24** | 每 band ~30 行，6 bands/GPU | topk 单次 ~370ms |
| **SUB_BATCH_DAYS=23** | 92 天分 4 批 | 控制峰值显存 |
| **4 卡线程并行** | 4 线程共享 data 零拷贝 | 零启动开销 |
| **无 .copy()** | numpy 切片连续，torch.from_numpy 零拷贝 | 省 ~50ms/band |

### 参数说明

| 参数 | 值 | 影响 |
|------|-----|------|
| WINDOW_HALF | 5 | 滑动窗口半径（固定，变影响精度） |
| N_BANDS | 24 | 纬度分带数，每 GPU 6 bands |
| SUB_BATCH_DAYS | 23 | 天数子批次，topk 峰值显存控制 |
| MAX_READ_WORKERS | 16 | ProcessPool 模式用（MPI 无效） |
| IO_FILES_PER_TASK | 16 | ProcessPool 模式用（MPI 无效） |

### 关键发现（profiling 结果）

每个 band（30行，4 sub-batches）耗时分解：

| 阶段 | 每 sub-batch | 每 band | 占比 |
|------|-------------|--------|------|
| topk(P90) | 373ms | 1492ms | **71%** |
| copy+unfold | — | 310ms | 15% |
| nanmean(clim) | 40ms | 160ms | 8% |
| GPU→CPU+numpy | 29ms | 116ms | 5% |
| numpy assign | 6ms | 24ms | 1% |

**结论**: compute 瓶颈是 topk（单 GPU 算力限制），调参无法突破。

## 性能演进

| 版本 | Compute | I/O | Total | 加速 |
|------|---------|-----|-------|------|
| 逐像素 CPU (原始) | 数小时 | — | 数小时 | 1x |
| f32 4卡 线程并行 | 37s | 29s | — | ~50x |
| + topk P90 + f16 | 16s | 15s | ~32s | ~60x |
| **+ MPI I/O + f16全程** | **14.9s** | **7.5s** | **~24s** | **~70x** |
| + 双节点半赛季 | ~8s | 7.5s | ~15s | 目标 |

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

## 下一步

- [x] MPI I/O + float16 全程计算，Total ~24s
- [x] Profiling 确认 compute 瓶颈在 topk（71%），无法进一步优化
- [ ] 双节点各跑半个赛季（152-197 / 198-243），无数据交换，Total 目标 ~13s
