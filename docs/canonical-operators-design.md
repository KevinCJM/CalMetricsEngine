# 第一阶段：118 个 Canonical Operator 的 C++ 注册表

## 1. 需求、范围与基线

本次交付：完整的 118 项 canonical 名称覆盖、真实可执行 C++ 数值实现、单一注册表、严格零拷贝输入、可复用 out/workspace、可移植 SIMD、契约/数值/内存测试，以及 wheel 验收。

不交付：DSL/AST 解析器、通用 DAG 执行器、全图 CSE/liveness、统一调度器、进程池或线程池。不修改 BetterSaaTaa，不替换其生产 NJIT 路径，不自动提交/推送/发布。

基线：CalMetricsEngine `cd2791e`；BetterSaaTaa HEAD `03530df`，仅读取 `backend/cal_indicators/`。外部仓库的其他工作区变更不属于本需求。

118 个名称以 `typed_numba_kernels.py::CANONICAL_OPERATOR_IDS` 为准；该文件 SHA256 为 `757251922914384fb4fdb295036ad2550179107d4ca14e88b211cca521e104fb`。源文件哈希、实际执行生成的 oracle 和全量名称列表保存在 `tests/data/canonical_reference.json`。后续源码变化不得无审核覆盖 fixture。

### 四问

1. 是否属于当前需求？C++ 算子、注册契约、绑定、测试、构建和说明属于；业务中心、服务部署、AST/DAG 迁移不属于。
2. 不改是否阻塞？没有统一 shape/dtype/status/out/scratch 契约会阻塞算子安全复用；完整 Scheduler 不阻塞本阶段，故不实现。
3. 能否最小改造？复用唯一 `_native` 扩展、CMake、现有 finance API 和测试；新增独立 `operators` 命名空间，不改历史 finance 金融口径。
4. 是否改变语义？数值默认值、缺失处理、同值边界和操作顺序取自源 NJIT/受控 lowering。允许本阶段明确的内存契约改善：`lag`、`transpose` 和二维 `diag` 默认返回只读视图而非数据副本；显式 `out` 时才写入结果。

## 2. 分层设计

```text
calmetrics_engine.operators
    catalog / get / call / named operators / Workspace
                         ↓
同一个 _native 扩展中的 operators 绑定
    ndarray/标量/状态转换、owner 保活、out 校验、GIL
                         ↓
C++ Operator Registry（唯一名称、opcode、arity、类别、策略真相）
    prepare → output shape + scratch requirements
    execute → 纯 C++ 函数
                         ↓
通用 elementwise / reduction / sequence / matrix / state 内核
    scalar / NEON / SSE2 / 按 CPU 与 OS 能力选择 AVX2
```

C++ 对外提供 `lookup`、`prepare`、`execute`。Python 仅做名称导出、参数绑定和对象封装，不做数值运算或逐元素 fallback。将来 DAG 执行器直接使用同一 C++ 注册表，无需重新穿过 Python。

## 3. 注册表和算子粒度

固定显式 opcode 1–118 与基线 canonical 名称顺序对应，今后只能追加，不能因排序/分组改变已分配值。它们属于 `canonical-native-1` 命名空间，不等于旧 Numba BASIC_OPCODES 或旧计划编号。注册表冻结、不开放动态重复注册接口，拒绝未知名称、未知 opcode 和不支持的参数。

原生 `Spec` 及绑定导出的 catalog 暴露：id、opcode、native registry 版本、family、最小/最大参数数、按 arity 区分的源参数名称、默认规则、shape 规则、输入/缺失策略、SIMD 能力、并行策略、status 契约以及组合依赖。`Operator.requirements(*args)` 在具体参数已知后返回输出 shape/kind、借用视图标记和 scratch 需求，避免用笼统 catalog 代替真实 shape 校验。

**118 个名字不等于 118 份独立数学实现。**

- 元素运算按数学语义和 operand 形状复用模板。
- 全轴/时间轴/资产轴 reduction 复用一个 reducer，禁止多处各写一套方差。
- `total_return` 等现有组合名称保留并声明 composition；C++ 入口调用共享的标量/扫描/reduction 语义，允许无临时数组的等价融合，但不是新建不可解释的业务黑盒。
- `linear_fit` 是一次带截距 OLS 求解，五个投影只读同一状态；其 slope/intercept/R²/标准误兼容名称复用该拟合实现。
- 回撤区间是四字段状态；三个投影不再次遍历数据。
- `rolling_window` 和 `rolling_apply` 是后续 compiler 控制节点，不在这 118 项中，不伪装成已实现 standalone 算子。

## 4. 输入、输出及工作区

### 类型

数值 ndarray 必须是本机字节序 float64；mask 为 uint8 0/1 或 NumPy bool（按一字节只读，不复制）。数值标量可以是 Python float/可精确表示的 integer。拒绝 list、object、隐式 float32 转换、非本机字节序、非法 rank 和 dtype 混用。

支持 rank 0/1/2 的相应重载；逻辑操作要求 mask 类型；仅标量广播，不做任意 NumPy 维度广播。相同 shape 不证明金融轴或计量单位一致——名义轴、价格基准和时点推导仍归后续 Typed DAG 层；本阶段 catalog 必须保留此边界。

### 零拷贝与布局

输入引用由绑定层持有至 C++ 调用结束。C++ 只借用指针、shape、以元素计的有符号 stride；读取 C/F order、负 stride、普通切片和 readonly ndarray 不作 contiguous 修复。

重型调用建议调用方在数据入口准备 product-major 连续数组，用零基半开 `[start,end)` 的基本切片传入某产品/区间。现有 finance API 的 inclusive 窗口接口不在本阶段改变。日期/位置由调用方保证轴正确；`days_between` 读取自然日编号，而不是未经转换的 Unix 纳秒。

SIMD 连续路径使用安全的非向量宽度对齐 load/store；dtype 对齐不等于 32/64-byte 对齐，窗口偏移也不保证 SIMD 宽度对齐。尾部不得越界 load，非单位 stride 使用安全 C++ 路径，禁止为了 SIMD 偷拷贝输入。

### out 和 scratch

数组算子支持 `out=`，shape/dtype 必须精确、可写且连续；不允许与任一输入内存区间重叠，防止污染 DAG 的共享数据。没有 out 时只分配一次最终输出。失败时显式 out 可能已有部分写入，调用方必须丢弃失败结果；不为事务保证复制整个 out。

`Workspace` 由调用方独占，可跨顺序调用复用。并发复用同一 Workspace 明确拒绝，不阻塞持有 GIL 的线程。没有传入时使用调用私有工作区。暴露需求与实际使用字节数，数值内核不持有全局可变 scratch。

quantile/median 为保持输入只读，会将被选中的数值一次复制到连续 `double` scratch 后排序；这属于显式算法工作区，可改善 cache locality，并通过 `algorithm_copy_bytes` 精确报告，不属于输入绑定层复制。`solve` 同样在算法工作区中初始化可修改系数矩阵/RHS。边界复制与算法工作区复制必须分开报告。

视图结果只读并持有原 owner；不得返回指向已销毁 workspace 的结果。普通输出由 NumPy 拥有，不随 workspace 复用失效。

绑定会沿可知的 ndarray/buffer owner 链校验可访问范围，拒绝可识别的越界 as_strided；不透明 capsule/exporter 仍需保证其指针和真实分配范围有效。原生调用者必须自行满足同样的输入有效性、输出不重叠和 Workspace 独占契约。

## 5. 必须保持的数学契约

- 普通 `variance/std` 默认 ddof=1，显式 ddof 必须为有限非负整数且小于样本数。
- `rolling_std` 默认 ddof=0，min_periods 默认 window；有限值计数，不压缩日期轴。
- quantile 使用线性插值，概率严格在 `(0,1)`；median 支持奇偶长度。
- 普通 reduce 保留顺序 Welford/源归约顺序，不为 SIMD 启用全局 fast-math；不擅自用 sum-of-squares 替代。
- mask 只选择有效输入位置，不将 NaN 乘以 0 当作清理；未选值不得污染结果。
- `lag(x,p)` 是原始契约的前缀 `x[:n-p]`，不是自动补 NaN 的等长移位；difference 输出 n-p。
- `drawdown_series` 输入正值 level path，输出有符号回撤；不同于历史 finance API 从 returns 构造路径。
- `new_high_mask` 首项为真，只有严格创新高为真。
- `last_drawdown_interval` 同深度选择最后谷底；无事件 status=0，无效输入 status=-1，未恢复为 NaN，不能伪造日期。
- `value_at` 非法/缺失位置返回 NaN；`days_between` 非法日期或倒序返回 NaN。
- `divide` 保持严格除零错误；`divide_or_default` 使用 1e-12 分母阈值且非有限输入仍输出 NaN。不能根据输出 shape 猜测 series 模式并更改 divide 语义。
- OLS 保留 0..n-1 隐式观察轴、截距、RSS/TSS 和样本数；投影不重新拟合。
- solve 保留部分主元和 1e-14 奇异阈值；covariance/correlation 明确沿 time 轴、ddof=1。
- 组合 annualized_return 保留受控 lowering 的运算顺序与 domain guards，不能擅自消去 `(product-1)+1`。

## 6. SIMD 与并发边界

本阶段实现 17 个连续元素/比较/mask 算子和 matmul 的真实 SIMD 及可用硬件选择，保留 scalar、ARM64 NEON、x86 SSE2，AVX2 只在独立编译目标和运行时 CPU/OS 检测均通过时启用。不能全局 `-march=native`、`-mavx2` 或 fast-math。

矩阵采用 4 行×2 SIMD 向量的寄存器分块，沿 k 顺序累计、分离乘加、不打包复制输入。右矩阵沿列单位 stride 且尺寸足够时使用该路径；行/列尾部安全标量处理。该选择保证本阶段零打包和数值契约，不代表已达到系统 BLAS 性能。

所有不适合向量化的递推、排序、状态和严格顺序归约仍在 C++ 中运行；这是合法 native scalar lane，不是 Python fallback。只报告实际实现/实际运行的 lane，不以函数后缀或候选清单证明 SIMD。

基础算子默认单线程，不创建每节点线程池。GIL 在数值执行期间释放；调用方已有线程可并发执行，workspace 独占、输入只读。统一 Scheduler/持久线程池/进程池留在之后阶段，不能因规模大而自动叠加并发。

## 7. API 设计

```python
from calmetrics_engine import operators as op

op.catalog()                         # 118 项，由 C++ 注册表生成
op.get('std')                        # 已解析算子 handle
op.call('std', values, 1.0)          # 可直接调用
op.std(values, ddof=1.0)              # 命名入口，仍委托同一 native handle
op.add(left, right, out=output)
result, audit = op.add(left, right, simd='auto', audit=True)
workspace = op.Workspace()
result = op.quantile(values, 0.95, workspace=workspace)
```

核参数按契约位置/名称绑定，包含一输入拟合 `values` 与双输入拟合 `x/y`、一输入协方差 `asset_returns` 与双输入 `lhs/rhs` 的区别。状态保持四/五元素 tuple，与源数值契约兼容；本阶段不引入完整 DSL 状态类。

`out/simd/workspace/audit` 为保留执行参数。审计返回本次调用的实际 lane、向量化元素数（matmul 为输出单元数）、边界复制量、算法 scratch 和必要初始化复制量，不使用可竞争的全局 last-call 状态。

## 8. 验收

1. 注册表名称集合和固定 opcode 与 118 名称 fixture 完全一致，所有项真实调用成功，未知项失败。
2. oracle 由当前源 NJIT 或明确的源组合 lowering 执行产生，不由新 C++ 生成；记录源哈希。fixture 测试不依赖外部仓库、Numba 或网络。
3. 比较全部算子、重载、默认参数、矩阵轴、状态投影；单独覆盖每个边界/异常契约。NaN 和 signed-zero 行为单独断言。
4. C/F/strided/negative stride/readonly 输入、输入未修改、视图生命周期、out 地址不变、alias 拒绝、workspace 并发/复用、共享内存视图。
5. scalar/SIMD 对照、被选择的 ISA 确认可执行；独立 C++ tests 和 ASan/UBSan。
6. 原 220 项回归不变通过；从 sdist 构建 wheel 并安装真实产物测试，检查新头文件和 Python 入口完整。
7. 可重复 benchmark 分开记录 scalar/SIMD、contiguous/strided、尺寸、热调用和内存；性能无证据不写“更快很多”。未在当前硬件运行的平台只称已配置。

## 9. 官方依据

- pybind11 NumPy/buffer/noconvert：https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
- NumPy ndarray strides 与内存布局：https://numpy.org/doc/stable/reference/arrays.ndarray.html
- Clang 多目标编译：https://clang.llvm.org/docs/LanguageExtensions.html#function-multiversioning

实际实现、完整算子表和测试结果另见 `canonical-operators-acceptance.md`；该文件在验证后填写，不预先宣称通过。
