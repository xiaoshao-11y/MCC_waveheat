# 海温气候态计算 — DCU GPU 加速

## 项目背景

MCC 海洋计算挑战赛，基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)。算法为 11 天滑动窗口，30 年基线。

## 当前阶段

**已完成所有优化，4 卡并行 GPU 加速，总耗时 ~57s (1.0 分钟)。**

| 阶段 | 状态 |
|------|------|
| 算法验证 | ✓ 与 float64 参考一致，RMSE < 0.001°C |
| float32 单卡 | ✓ 4.1 分钟 |
| 4 卡线程并行 | ✓ 2.8 分钟 |
| I/O 优化 + 重叠 | ✓ 1.9 分钟 |
| topk P90 + 去inode + 后台import | ✓ 0.9 分钟 |
| float16 存储 + 节点优选 | ✓ **~0.7 分钟（冷启动，最佳节点）** |
| 精度验证 | ✓ verify_output.py 通过 |
| 自动空闲节点选择 | ✓ get_climatology.sh |

### 最终性能（冷启动）

```
Setup:    24-40s ← import torch + HIP 初始化（后台重叠，被 I/O 隐藏）
I/O:      29-49s ← 读 3060 个 .nc 文件 (Lustre 共享存储，节点间波动)
Compute:  13-14s ← 4 张 DCU Z100 并行，topk P90
Save:      1-2s ← 写 92 个 NetCDF
Total:    45-66s (0.7-1.1 min)（取决于节点）
```

Setup 在后台线程与 I/O 并行重叠，wall time = max(Setup, I/O)。小计 = I/O + Compute + Save ≈ 44-66s。

冷启动最佳节点（f16r4n02）：I/O 29s，总时间 45s。其余节点 57-66s。

## 加速方法

### I/O 优化（ProcessPoolExecutor, 24 workers）

| 优化 | 原理 | 效果 |
|------|------|------|
| **netCDF4 直读** | `Dataset(fp)["data"][:]` 代替 xarray，跳过元数据解析 | 大幅减少单文件开销 |
| **year-grain pool.map** | 24 进程，每进程读一整年 102 个文件，减少调度开销 | 减少进程切换 |
| **预建路径字典** | 启动时一次 glob 建好 `{YYYYMMDD: path}`，读文件直接查表 | 零 `os.path.exists/isfile/join` |
| **去掉 inode 排序** | 之前 3060 次 `os.stat()` 打爆 Lustre 元数据服务器 | **65s → 33s** |
| **float16 存储** | numpy 数组 float16，I/O 传输量减半（5.92 GB） | **33s → 22-31s** |
| **HDF5_USE_FILE_LOCKING=FALSE** | Lustre 只读场景关闭 HDF5 文件锁 | 减少锁竞争 |

### Setup 优化（import torch + HIP 初始化）

| 优化 | 原理 | 效果 |
|------|------|------|
| **后台线程 import torch** | `threading.Thread` 执行 import + HIP 检测，与 I/O 并行 | 被 I/O 完全隐藏 |
| **PYTHONPYCACHEPREFIX=/tmp/pycache** | .pyc 缓存到本地 SSD，避免每次从 Lustre 编译；/tmp 满时回退 ~/.cache | **~30s → ~20s** |
| **跳过 conda activate** | 直接 `PATH=env/bin:$PATH`，省去 conda 初始化 | 省 2-3s |

### Compute 优化（GPU 计算）

| 优化 | 原理 | 效果 |
|------|------|------|
| **float16 存储 + float32 计算** | 存储用 float16（5.92 GB CPU），GPU 上转为 float32 计算 | 内存减半，精度无损 |
| **topk P90** | `torch.topk(k=34)` 只排最大 ~10%，替代 330 元素全排序 | **29s → 19s** |
| **torch.unfold() 零拷贝窗口** | 不复制数据生成 11 天滑动窗口视图 | 省 ~10 GB 显存 |
| **torch.inference_mode()** | 关闭 autograd 计算图追踪 | 省显存 |
| **SUB_BATCH_DAYS=23** | 92 天分 4 批，减少循环和中间张量分配 | 降低峰值显存 |
| **in-place 运算** | `mul_().add_()` 原位计算，不分配中间张量 | 省显存 |
| **4 卡线程并行** | 4 线程共享 `data_np` 零拷贝，各绑一张 GPU | **37s → 13s** |
| **del + empty_cache** | 每 band 后释放 GPU 显存 | 防 OOM |

### Shell/环境优化

| 优化 | 原理 | 效果 |
|------|------|------|
| **ulimit -c 0** | 禁用 core dump | 防磁盘被 4-8GB core 文件占满 |
| **优先选 f16 节点** | f17 节点 Setup 极慢（~60s），自动过滤 | 避坑 |
| **优先选 f16r4n02** | 已知 I/O 最快的节点（冷启动 31s），优于其他 f16 节点（40-49s）| 冷启动最优 |
| **BAD_NODES 排除** | 手动排除已知慢节点 | 灵活 |

### 关键：并行重叠

```
Setup (后台线程)  ←── 真正并行 ──→ I/O (主线程, ProcessPoolExecutor)
                   wall = max(Setup, I/O) ≈ I/O
                   不是 Setup + I/O
```

最终耗时 = **max(Setup, I/O)** + Compute + Save ≈ **I/O + 13 + 2 ≈ 46-66s**（取决于节点）。

## 性能演进

| 版本 | Compute | Total | 累积加速 |
|------|---------|-------|----------|
| 逐像素 CPU (原始) | 数小时 | 数小时 | 1x |
| f64 GPU 向量化 | ~200s | ~6.6min | ~20x |
| f32 单卡 | 99s | ~4.1min | ~32x |
| f32 4卡 线程并行 | 37s | ~2.8min | ~50x |
| + netCDF4直读 + 重叠 | 29s | ~2.1min | ~50x |
| + topk P90 + 去inode + 后台import | 19s | ~0.9min | ~60x |
| **+ float16 + 节点优选** | **14s** | **~0.7min** | **~60x** |

## 下一步

- [x] 所有优化完成，冷启动总时间 ~45s（最佳节点 f16r4n02）
- [ ] 无更多可优化方向，I/O 受限于 Lustre 物理带宽
