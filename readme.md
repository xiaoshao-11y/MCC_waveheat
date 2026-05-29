# 海温气候态计算 — DCU GPU 加速

## 项目背景

MCC 海洋计算挑战赛，基于 1991-2020 年全球海表温度 (SST) 数据，计算夏季 (6-8 月) 气候态均值 (Climmean) 和 90% 分位阈值 (P90_sst)。算法为 11 天滑动窗口，30 年基线。

## 当前阶段

**已完成所有优化，4 卡并行 GPU 加速，比赛计时 ~32s。**

| 阶段 | 状态 |
|------|------|
| 算法验证 | ✓ 与 float64 参考一致，RMSE < 0.001°C |
| float32 单卡 | ✓ 4.1 分钟 |
| 4 卡线程并行 | ✓ 2.8 分钟 |
| I/O 优化 + 重叠 | ✓ 1.9 分钟 |
| topk P90 + 去inode + 后台import | ✓ 0.9 分钟 |
| float16 存储 + 节点优选 | ✓ ~0.7 分钟（冷启动，最佳节点） |
| h5py + 共享内存（失败） | ✗ HDF5 全局锁串行化，反而更慢 |
| 计时修正 + 节点排名 | ✓ real ~29s 不含 setup |
| Setup前置 + Validation后置 | ✓ Setup 移到 I/O 前，Validation 移出计时 |
| 精度验证 | ✓ verify_output.py 通过 |
| 自动空闲节点选择 | ✓ get_climatology.sh |

### 最终性能（冷启动，最佳节点 f16r4n13）

```
Prep:       0.2s  (file index + format probe, 不计时)
Setup:      14s   (import torch + HIP 初始化, 不计时)
─── 比赛计时起点 ───
I/O:        15s   (3030 files, 5.92 GB, 24 workers, netCDF4 直读)
Compute:    14s   (4×DCU Z200 并行, topk P90)
Save:        1.5s (92 个 NetCDF 输出)
─── 比赛计时终点 ───
Total:      32s   (I/O+Compute+Save+Other, 比赛计时)
Shell real: 51s   (完整进程, 含 import + prep + setup)
Validation: 0.4s  (计时后)
```

### 实测节点排名（24 workers）

| 节点 | I/O | real (no setup) | Shell real |
|------|-----|-----------------|------------|
| f16r4n13 | 13.5s | **28.9s** | 46.8s |
| f16r4n03 | 14.4s | **29.9s** | 49.1s |
| f16r4n12 | 14.6s | **28.9s** | 50.6s |
| f16r4n02 | ~29s | ~45s | ~67s |

节点优先策略：`f16r4n13 > f16r4n03 > f16r4n12 > f16*`

## 加速方法

### I/O 优化（ProcessPoolExecutor, 24 workers）

| 优化 | 原理 | 效果 |
|------|------|------|
| **netCDF4 直读** | `Dataset(fp)["data"][:]` 代替 xarray，跳过元数据解析 | 大幅减少单文件开销 |
| **year-grain pool.map** | 24 进程，每进程读一整年 102 个文件，减少调度开销 | 减少进程切换 |
| **预建路径字典** | 启动时一次 glob 建好 `{YYYYMMDD: path}`，读文件直接查表 | 零 `os.path.exists/isfile/join` |
| **去掉 inode 排序** | 之前 3060 次 `os.stat()` 打爆 Lustre 元数据服务器 | **65s → 33s** |
| **float16 存储** | numpy 数组 float16，I/O 传输量减半（5.92 GB） | **33s → 14s** |
| **HDF5_USE_FILE_LOCKING=FALSE** | Lustre 只读场景关闭 HDF5 文件锁 | 减少锁竞争 |
| **h5py 尝试（失败）** | 想减少 per-file NC 元数据开销，但 HDF5 C 库全局锁串行化多线程 | 反而变慢（51s） |
| **MAX_READ_WORKERS=30（失败）** | 增多 worker，但 I/O 瓶颈在 Lustre 连接质量而非 CPU | 无提升，反增 CPU 争用 |

结论：I/O 瓶颈是 Lustre 节点间连接质量差异（同一集群差 2 倍），不是 CPU/worker 数。

### Setup 优化（import torch + HIP 初始化）

| 优化 | 原理 | 效果 |
|------|------|------|
| **PYTHONPYCACHEPREFIX=/tmp/pycache** | .pyc 缓存到本地 SSD，避免每次从 Lustre 编译 | Setup 加速 |
| **跳过 conda activate** | 直接 `PATH=env/bin:$PATH`，省去 conda 初始化 | 省 2-3s |
| **Setup 前置（当前）** | Prep → Setup → I/O → Compute → Save，Setup 在计时起点之前 | 比赛计时纯净，不含 Setup |

> Setup (import torch + HIP) 在 I/O 之前执行，`total_start` 计时起点在 I/O 开始处。比赛要求环境加载不计时，Setup 置于计时窗口之外。

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
| **实测节点排名** | f16r4n13 > n03 > n12，基于实测 I/O 数据 | 选最快节点 |
| **BAD_NODES 排除** | 手动排除已知慢节点 | 灵活 |
| **{ time ... } 在脚本** | bash time 保留完整进程计时 | 与 Python 内部计时互补 |

### 计时说明

```
Shell time real: 完整进程（import + 所有）
Python TIMING:   I/O起点到结束的详细分解
Python real:     I/O + Compute + Save + Other（比赛计时）
```

> Prep 和 Setup 在计时起点之前，Validation 在计时终点之后，均不计入比赛时间。

## 性能演进

| 版本 | Compute | real (比赛计时) | Shell real | 加速 |
|------|---------|-----------------|------------|------|
| 逐像素 CPU (原始) | 数小时 | — | 数小时 | 1x |
| f64 GPU 向量化 | ~200s | — | ~6.6min | ~20x |
| f32 单卡 | 99s | — | ~4.1min | ~32x |
| f32 4卡 线程并行 | 37s | — | ~2.8min | ~50x |
| + netCDF4直读 + 重叠 | 29s | — | ~2.1min | ~50x |
| + topk P90 + 去inode + 后台import | 19s | — | ~0.9min | ~60x |
| + float16 + 节点优选 | 14s | ~45s | ~0.7min | ~60x |
| + 计时修正 + 节点排名 | 13s | ~29s | ~47s | ~60x |
| **+ Setup前置 + Validation后置** | **14s** | **~32s** | **~51s** | **~60x** |

## 下一步

- [x] 所有优化完成，最佳节点 f16r4n13 real ~29s
- [x] I/O 优化已到极限，瓶颈在 Lustre 节点间连接质量
- [ ] 持续发掘更快节点，更新排名
