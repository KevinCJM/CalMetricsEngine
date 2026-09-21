# 缺失数学能力补齐验收

日期：2026-09-21。设计与契约见[详细设计](mathematical-composition-design.md)。

## 变更范围

- 在唯一 C++ registry 追加 7 个独立算子，原 opcode 1–118 与历史数值参考保留。
- 扩展原编译器/执行器的分块、筛选、分组和有界求根作用域；指标保持为显式数学组合。
- 扩展原 Typed IR、数组边界、Planner 与 worker 传输，支持 int64 类别/索引、矩阵/向量和必要的工作空间。
- 所有运行时计算继续由 C++ 执行；没有引入 Python 数值回退，也没有修改研究平台的默认服务路径。

## 验证状态

当前源码对应的已安装 wheel 完成验收：

| 检查 | 实际结果 |
| --- | --- |
| Python 全量回归 | **4210 passed，0 failed** |
| C++ Release CTest | **6/6 套通过** |
| C++ Debug + ASan/UBSan CTest | **6/6 套通过** |
| 指标组合独立参考 | 26 项，已包含在全量回归中 |
| Typed 输入、传输、矩阵预算专项 | 31 项，已包含在全量回归中 |
| 新增测试静态检查 / diff 空白检查 | Ruff 与 `git diff --check` 通过 |
| 旧 118 算子固定参考数据 | `tests/data/canonical_reference.json` 未修改 |

原生和 Python 回归都运行了真实本地 process/shared-memory 路径。普通沙箱禁止 `shm_open`，完整验收在允许创建测试共享内存的本地执行环境完成；未跳过共享内存用例。

### 数学与工程覆盖

- EMA 偶数/奇数周期、缺失与区间重置；KDJ 首行和两层共用更新 mask；TRIX 三层递推；ADXR 派生序列移位。
- Hurst 原定义与 RS 定义的分块、筛选顺序和退化样本；CPR 先分块再删空的历史口径。
- 分组集中度、超过 2^53 的整数 ID 去重、稳定排序/同步 gather、显式 BS 公式与 IV 二分反解。
- 空集合、非法索引/mask、求根不夹根/不收敛、组内错误隔离、嵌套作用域资源上限。
- 矩阵/向量跨线程与进程传输、readonly/负 stride、prepared 复用、底层 owner 变更、矩阵增长与嵌套工作区预算。
- `transpose`/`diag` 借用链的 arena 生命周期；静态资产分组结果保留资产轴，禁止冒充时间序列根。

### 原有性能门禁

同机配对暖执行，CPU budget=8；大批次各 5 轮、微任务各 201 轮。大批次为 2520 条历史 × 12 区间 × 16 指标；微任务为 1 产品 × 1 区间 × 5 指标。

| 工作负载 | C++ / NJIT 时间 | 比值 | 门限 |
| --- | --- | --- | --- |
| 500 产品 | 30.400 / 47.473 ms | **0.640** | ≤0.90 |
| 1000 产品 | 57.304 / 96.198 ms | **0.596** | ≤0.90 |
| 63 条微任务，prepared | 3.041 / 5.041 μs | **0.603** | ≤0.90 |
| 252 条微任务，prepared | 7.041 / 9.791 μs | **0.719** | ≤0.90 |
| 63 条微任务，普通 Scheduler | 4.708 μs / 对应配对 NJIT | **0.934** | ≤1.00 |
| 252 条微任务，普通 Scheduler | 8.667 μs / 对应配对 NJIT | **0.874** | ≤1.00 |

四组门禁全部通过。基准中 single/thread/process/auto 的结果与 NJIT 最大绝对误差均为 0；审计确认 `python_fallback=0`、`python_worker_callbacks=0`，整次请求只跨一次 Python→C++ 边界。该结论针对既有基准，不能推导所有新指标均具有相同加速比。

进程基准复用预先创建的共享输入，一次性复制不计入重复计时；每次区间描述和输出共享区创建仍在计时内。保留工作区容量、必要复制与基线 RSS 记录，不将同进程累计峰值误称为独立内存节省比例。

### 构建身份与复现

- 环境：macOS 15.6.1 / arm64，CPython 3.12.11，AppleClang 17.0.0，NumPy 1.26.4，Numba 基线 0.60.0。
- 引擎包：`calmetrics_engine 0.3.0`；验收时源码基于 `f952f37` 并包含待提交变更，因此以 build ID 标识本次构建，不能仅用当时的 HEAD 表示。
- 引擎 build ID：`b4392aed8bbff8c1f12eb63c5c46947e10f8ede8d0e4c73641ce82d980979b91`，已独立按当前 C++ 源码、CMake、包元数据及编译参数重新计算并匹配。
- wheel SHA-256：`71fc2cb781ce18db79c704f5ae25033e9e39b5d2d6b14dc79dc623300d056706`。
- BetterSaaTaa 基线 HEAD：`c105eb10b2013abf5f670953ea2bf162b865f03b`，计算模块无本地修改；仅作为测试参照，不成为生产包依赖。

[机器可读验收证据](mathematical-composition-evidence.json) 保存命令、构建身份、测试/基准脚本哈希、各执行路径计时、数值差异、容量与复制审计。

```bash
PYTHONPATH=<installed-wheel> python -m pytest tests -q --tb=short -p no:cacheprovider
ctest --test-dir <release-build> --output-on-failure
ctest --test-dir <debug-sanitized-build> --output-on-failure
PYTHONPATH=<installed-wheel> NUMBA_CACHE_DIR=<temporary-cache> \
  python tools/check_phase2_performance.py --better-root <BetterSaaTaa-checkout> \
  --products 500 1000 --history 2520 --intervals 12 --micro-histories 63 252 \
  --cpu-budget 8 --repeats 5 --output-dir <temporary-output>
```

`<...>` 为使用者本地构建/安装目录占位符；sanitized 构建使用 `CALMETRICS_ENGINE_SANITIZE=ON`。未在 Linux、Windows、其他 CPython 或 x86 ISA 上运行本轮验收。

## 固定证据边界

- 原能力审计及 333 条清单保留为变更前快照；本次新增能力不意味着全部指标定义已经迁移。
- 指标组合参考代码仅存在测试中，生产包不依赖 MetricsFactory 或其目录。
- 真实市场数据关联、合约配对、复权/PIT、前端目录和业务发布门禁属于调用方集成，未在本次认证。
- `numeric_arena_bytes` 等容量审计可能包含线程缓存此前保留的空间；不能将缓存容量与本次计划的增量需求直接作相等比较。
- 必要排序索引、离散筛选和 gather 输出分配单独计入原生工作区/复制证据，不声称全链路零分配。
