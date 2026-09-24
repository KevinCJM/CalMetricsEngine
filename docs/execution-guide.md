# 执行与性能指南

[文档导航](README.md) · [使用手册](user-guide.md) · [算子参考](operator-reference.md)

本页说明当前 Planner 的行为，不承诺任意数据上并行都更快。依据是 [planner.cpp](../cpp/planner.cpp)、[默认配置](../cpp/include/calmetrics_engine/planner.hpp)、[原生运行时](../cpp/native_runtime.cpp) 与 [async 适配](../src/calmetrics_engine/runtime.py)。

Excel 下载是独立的按需任务：[公式导出](excel-export.md) 用 C++ 规划、串行流式写出，不沿用 arena 槽位覆盖 Excel 中间步骤，也不自动启动 Excel。补算 C++ 参考值使用现有 Scheduler 的单任务准入；常规计算不规划、生成或重算 Excel。表格引擎自己的重算线程由其管理，不能当作 C++ Planner 的并行证据。

## 1. 五个概念并不互斥

| 能力 | 解决的问题 | 如何启用 |
| --- | --- | --- |
| SIMD | 一个 CPU 核的一条指令处理多个数值 | 内核、CPU/OS、布局与长度满足条件时派发 |
| 多线程 | 同一进程分担独立计算块 | Planner 选择 thread，统一 C++ ThreadPool 执行 |
| 多进程 | 原生计算隔离、重负载分块、Hard Stop | Planner 选择 process，启动/复用 calmetrics_worker |
| 协程 | 调用方事件循环不被同步请求阻塞、并发等待请求 | 显式调用 execute_async/execute_many_async |
| 共享内存 | 减少跨进程重复搬运，管理共享输入/输出 | process 计划满足运输条件，或显式建立 SharedInputBundle |

可以同时使用“协程提交 → 多进程执行 → 每个进程单计算线程 → 内核 SIMD → 共享内存”。目前不启用每个进程再套一组计算线程的混合并行。

## 2. SIMD：什么情况下真正使用

当前 canonical 目录标记 18 个显式 SIMD 算子：add、subtract、multiply、divide、minimum、maximum、negate、absolute、sqrt、reciprocal、equal、not_equal、less_than、less_equal、greater_than、greater_equal、finite_mask、matmul。

- ARM64 使用 NEON；x86_64 基线 SSE2，AVX2 还要求独立目标已构建且 CPU/OS 支持。当前无 AVX-512 实现。
- 元素路径需要符合内核要求的连续布局和足够长度；尾部用安全标量计算。int64 比较等特定重载不因算子总标签而保证 SIMD。
- matmul 要求右矩阵列方向单位步长及足够分块大小，保持内积累加顺序；不为了 SIMD 隐式打包输入。
- 非单位步长、小数组、不支持的重载、顺序敏感的归约/递推可能走 C++ 标量路径；仍然是原生计算。
- 直接调用可用 simd="auto" 或 "scalar"；显式要求不可用 ISA 会报错。NaN 处理不因 SIMD 而改成自动删行/填零。
- Planner 的 simd_min_elements=128 影响计划中的候选节点标记，**不是所有内核统一的硬切换阈值**。实际向量化情况看审计，不仅看 simd_nodes 或 simd_eligible。

```python
import numpy as np
from calmetrics_engine import operators as op

x = np.arange(256, dtype=np.float64)
y, audit = op.add(x, 1.0, simd="auto", audit=True)
np.testing.assert_array_equal(y, x + 1.0)
print(op.available_simd())
print({key: audit[key] for key in ("isa", "vector_elements", "input_copy_bytes")})
```

不能从“C++”或“SIMD 可用”推导一定超过系统 BLAS。矩阵、递推、排序、窗口需在目标机器按真实端到端负载测量。

完整逐元素 typed 图还可以使用分段寄存器融合：保持逻辑 DAG，将相邻的两个标量广播加/减/乘及可选条件组合为已有 AOT SIMD 内核。长链可分成多个区域，共享中间值仍按消费者需求保留；其余安全逐元素节点继续在缓存块内执行。泛用中间块复用已有生命周期槽，整图短链减少重复描述及逐块结果计数。该路径保留逐次几何/参数校验、独立结果、负步长和错误契约；不代表任意子图、窗口或矩阵计算都已融合。设计、前后实测与尚未通过的 NJIT 门禁见[物理执行优化记录](physical-execution-optimization.md)。

## 3. Planner 的选择顺序

这里的 rows 是**请求的区间条数**，不是观察点数 T；work 是 lowering/CSE/融合后物理 DAG 的结构性成本估计，不是 FLOPs、毫秒或节点数。

1. 校验输入、区间、dtype/轴、图身份，估算输入/输出/状态及 scratch 容量。
2. 若 hard_stop=True，选择 process。否则同时满足 `rows >= max(2,min_rows_per_worker)`，以及工作量达到 process_work_units **或**输入字节数达到 process_input_threshold_bytes，才走普通 process。
3. 未选择 process 时，若为标量根图、`cpu_budget>1`、`0<rows<cpu_budget`，存在至少两个足够重的独立编译分支且总成本达到 dag_branch_work_units，则考虑 thread 的 dag_branch 计划。共享上游与融合组先合并，不按每个节点开线程。
4. 单区间的完整逐元素 typed 图，在输入为连续 float64、数组形状相同、每个线程至少一个完整 2048 元素块，且每线程工作量达到 thread_work_units 时，可选择 thread 的 tensor 分块；所有输出共享同一分块内的上游计算。
5. 否则，当 `work>=thread_work_units`、`rows>=2` 且 `cpu_budget>1`，选择普通 thread。
6. 其余为 single，在调用者的原生线程计算。
7. 内存预算可降低 worker 数量；降至 1 仍超预算则失败，不隐式修改算法或精度。当前 CPU 准入还可能使请求排队；计划 worker 数不等于任意时刻真正并发数。

普通进程数初值为 `min(cpu_budget,max_processes,max(1,rows/min_rows_per_worker))`（整数除法）；普通线程数初值为 `min(cpu_budget,rows)`。Hard Stop 即使仅一个区间、一个 CPU 也需要进程隔离。单区间普通时序根不自动获得跨时间递推并行。

分块优先考虑 product_ids 提供的足够产品组，否则按区间；使用物理成本加权，而不是简单按行均分。DAG 分支是区间并行不足时的受限补充。目前仅 scalar 根适用 dag_branch 策略；tensor 分块只适用于上述可证明逐元素独立的图，不能沿递推时间轴拆分。

### 当前默认值

| PlannerConfig 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| thread_work_units | 250000 | 普通线程成本门槛；也参与重分支判定 |
| dag_branch_work_units | 5000000 | 独立 DAG 分支并行总成本门槛 |
| process_work_units | 4000000000 | 普通进程成本门槛 |
| process_input_threshold_bytes | 268435456（256 MiB） | 普通进程输入大小门槛 |
| shared_memory_threshold_bytes | 8388608（8 MiB） | 进程共享输入运输门槛 |
| min_rows_per_worker | 16 | 进程准入/数量与分块相关参数；不是每个观察窗口的长度 |
| simd_min_elements | 128 | Planner SIMD 候选标记长度 |
| max_processes | 8 | 进程数上限，还受 CPU/内存预算限制 |
| max_async_jobs | 8 | 每次 execute_many_async 的并发桥接上限 |

这些配置可通过 PlannerConfig 构造指定。改变阈值需实测，不建议照抄测试中的极小门槛作为生产默认。cpu_budget 是 CPU 并发上限，不是承诺一定启动该数量的 worker。

### 查看计划，再执行

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler, PlannerConfig

graph = GraphCompiler({"x": "series"}).compile("mean(x)")
x = np.arange(32, dtype=np.float64)
start, end = np.array([0, 16], np.int64), np.array([16, 32], np.int64)
# 教学示例：降低线程门槛以观察 thread；不代表该小样本并行更快。
config = PlannerConfig(thread_work_units=1, process_work_units=1e100)
with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
    plan = scheduler.plan(graph, {"x": x}, start, end)
    print(plan.metadata()["reason_codes"])
    result = scheduler.execute(graph, {"x": x}, start, end, plan=plan)
    assert result.plan.lane == "thread"
np.testing.assert_allclose(result.values[:, 0], [7.5, 23.5])
```

同一进程内的图、直接算子和 cal_* 共用原生 CPU 准入。创建多个 Scheduler 不等于获得多个互不约束的 CPU 池；cal_* 的 n_threads 也只表示请求上限。

## 4. 进程、共享内存和复制边界

**自动共享运输条件**：已经选择 process，且满足 hard_stop、输入已是 SharedInputBundle/ModelPayload、输入字节数达到共享阈值，或任一分块的完整结果响应超过 256 MiB IPC 单帧上限。结果检查包含容量、逐元素状态、实际形状、根状态和协议头；不能只看输入大小。降低进程数后重新检查分块与内存预算。共享路径的状态/形状响应仍须在单帧上限内，否则规划阶段报错，调用方应缩小区间批次。单独超过 8 MiB 不会强制多进程，也不保证普通单线程请求新建共享内存。

小型 process 请求可使用显式计量的 inline IPC。大输入或 Hard Stop 路径由父进程持有共享映射、worker 只读附加；worker 将不同结果块写入不重叠区域。共享输出返回的 NumPy 视图持有原生 owner，无最终整块结果复制。typed 多输出按“区间→根”使用 8 字节对齐的容量槽，每个 worker 写互不重叠的区间块；每根实际形状和错误状态另行回传。Planner 计入容量、状态和描述信息，不能只用所有根的统一 dtype/长度估算。

| 环节 | 复制含义 |
| --- | --- |
| 普通 ndarray → 原生视图 | 符合入口契约时不复制 |
| 普通连续 ndarray → SharedInputBundle | 一次明确的共享存储初始化复制；多次调用可复用 |
| 小输入 inline IPC | 存在序列化/运输复制，不能宣称全路径零拷贝 |
| 分位数排序、线性求解 | 必要算法 scratch 初始化复制 |
| 共享输出 → NumPy 视图 | 原生 owner 保活，无最终数值数组复制 |
| statuses 结果/进程运输 | 独立计量，不能被数值零拷贝口号忽略 |

SharedInputBundle.from_inputs 要求各输入 C 连续；该共享构造入口的限制比 typed 矩阵普通绑定更严格。输入映射完成初始化后物理只读。不可在执行未结束时释放共享输入；返回的独立结果在关闭 scheduler/bundle 后仍有效。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler, PlannerConfig, SharedInputBundle

x = np.arange(32, dtype=np.float64)
graph = GraphCompiler({"x": "series"}).compile("mean(x)")
config = PlannerConfig(process_work_units=1, min_rows_per_worker=1,
                       max_processes=2)
# 仅演示可控进程/共享路径；不作为性能配置建议。
with SharedInputBundle.from_inputs({"x": x}) as bundle:
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        result = scheduler.execute(graph, bundle,
            np.array([0, 16], np.int64), np.array([16, 32], np.int64))
        assert result.plan.lane == "process"
        assert result.audit["use_shared_memory"]
        assert result.audit["output_copy_bytes"] == 0
np.testing.assert_allclose(result.values[:, 0], [7.5, 23.5])
```

worker 是随同 wheel 分发的独立 C++ 可执行文件，不启动 Python。协议版本必须匹配；不能从另一版本随意拷贝 worker。执行环境需要允许创建进程和共享映射，资源/worker 错误直接失败，不回退到 Python。

## 5. 协程：由调用方选择

execute_async 用 asyncio.to_thread 桥接阻塞的完整原生请求，数值循环不在事件循环里。execute_many_async 用 Semaphore 限制本次批次的桥接并发，并按输入任务顺序返回结果；原生 CPU 预算仍统一控制所有批次和同步请求。

```python
import asyncio
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

async def main():
    graph = GraphCompiler({"x": "series"}).compile("mean(x)")
    with AdaptiveScheduler(cpu_budget=2) as scheduler:
        jobs = [dict(graph=graph, inputs={"x": np.array([v, v+2.])},
                     starts=np.array([0], np.int64), ends=np.array([2], np.int64))
                for v in (1., 3.)]
        results = await scheduler.execute_many_async(jobs)
        np.testing.assert_allclose([r.values[0, 0] for r in results], [2., 4.])
        assert all(r.audit["async_orchestration"] for r in results)

asyncio.run(main())  # notebook / 已有事件循环中改为 await main()
```

异步适用于 Web 服务或需要并发等待的应用，不是数学提速开关。取消 awaiter 不会强杀已经进入原生计算的工作；输入所有权、scheduler 和资源必须保持到实际工作结束。

## 6. 超时、Hard Stop 与结果保存

普通线程超时是协作式边界检查，正在运行的长内核不能被安全强杀。需要硬截止使用 `execute(..., hard_stop=True, timeout=秒数)`；原生运行时采用隔离的一次性进程，失败/超时终止并等待 worker 后再释放资源。仅传 timeout 不等于自动要求硬杀线程。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

graph = GraphCompiler({"x": "series"}).compile("mean(x)")
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    result = scheduler.execute(graph, {"x": np.array([1., 2., 3.])},
        np.array([0], np.int64), np.array([3], np.int64), hard_stop=True, timeout=30)
    assert result.plan.lane == "process"
    assert result.audit["hard_stop"] and result.audit["use_shared_memory"]
assert result.values[0, 0] == 2.0
```

typed 的输出形状可随参数/数值变化，但必须落在计划容量内；prepared 复用时不缓存旧实际形状。typed run 返回带状态的 Result，取值见[使用手册](user-guide.md)。

prepared 当前只接受 single-lane 计划，是稳定输入/几何的重复执行入口，不是并行 execute 的通用替代或流式状态容器。需留存结果用 run_snapshot；借用输出会被覆盖，参见[使用手册](user-guide.md#9-重复执行与结果所有权)。

普通 execute 在同一引擎的输入绑定缓存中复用计划；每次仍复核输入布局、区间/产品内容及 CPU 限制。内存预算、Hard Stop 或异步策略变化时重新规划。普通入口的区间几何变化会重建计划；用户显式提供的旧计划和 prepared 的几何变化仍报错。计划复用不缓存结果，每次结果及执行审计均独立。

## 7. 如何判断本次执行是否符合预期

- plan.metadata()：lane、reason_codes、parallel_dimension、chunks/branch_tasks、预计 worker/内存和 simd_nodes。
- result.audit：实际 cpu_tokens、native_threads/native_processes、排队/计算耗时、复制字节数、共享输出所有权及 native_chunks。
- native_chunks 新增 fused_pointwise_calls、pointwise_tiles、vector_elements、root_copy_bytes、state_copy_bytes、direct_output_bytes 和 pointwise_workspace_capacity_bytes。向量计数仅包含显式 SIMD 实际处理的“算子／扫描 × 元素”总数，同一元素可计多次，不包含未经核实的编译器自动向量化；没有逐节点 ISA 明细，不能据此宣称每个候选节点均向量化。容量不是本次新分配字节数，direct_output_bytes 是计算写入量，不属于复制。
- engine_build_id、plan_fingerprint、registry/IR 版本和 input_dtypes/output_dtype 用于追溯。凭据不是外部认证协议。
- estimated_total_memory_bytes 是输入/输出/运输/arena/scratch 估计，不是 OS RSS 上限；峰值 RSS 另行测量。

性能排查顺序：先验证公式与区间 → 看复制和布局 → 看共享依赖是否复用 → 看计划分块/排队 → 最后调整阈值与 CPU 预算。重复固定任务考虑 prepared；需要保留结果时比较 run_snapshot，不能拿复用裸缓冲的耗时冒充独立结果成本。

完整性能门禁见 [AGENTS.md 第 16 节](../AGENTS.md#16-performance-evidence) 和 [门禁工具](../tools/check_phase2_performance.py)。本页示例验证功能路径，不提供性能提升或跨平台运行证明。


## 8. M0/M1 开发分支的执行边界

原生图协议升级为 7，worker 协议升级为 7；旧图定义通过重新编译获得新计划，worker 必须与当前 wheel 同构建。rank-3 的 time 首轴参与区间裁切，静态轴保持整块；空张量保持已知空形状。不同 dtype/shape 的具名输出继续共享一个带容量上界的结果缓冲。

ModelPayload 初始化有一次独立持久化复制，使用 8 字节对齐字段。Planner 计入包含对齐填充的载荷字节数；共享进程路径按一个模型区域复用只读映射，不能按字段重复计算同一映射字节数。请求/响应携带模型内容身份，父进程校验后接收结果。模型身份不是安全认证或金融适用性证明。

有界迭代的最坏迭代次数、子图工作量、双状态缓冲及诊断纳入计划。每轮检查计算预算与进程内取消/截止；强制硬停止仍使用隔离进程。通用解释执行的子图开销及中间复制必须在基准中计入，不能仅凭 C++ 实现宣称快于 NJIT。

新增协议负载由 `docs/platform-foundation-workloads.json` 固定，运行 `tools/check_foundation_performance.py --output <报告路径>`；旧四组门禁继续运行 `tools/check_phase2_performance.py`。新脚本保留配对原始耗时、单侧 95% 上界和隔离进程 RSS 样本；零增量或少于两页的内存优势标记未证实并失败。RSS 采样不能替代完整原生分配追踪，也不能代表所有平台算法已验收。最新状态见[开发设计](platform-foundation-design.md)。

## 9. 本轮物理执行优化与审计

完整逐元素 typed 图可融合 add/subtract/multiply/minimum/maximum/negate/absolute、数值比较和 finite_mask；数组需具有同一形状，标量可广播。不支持的 dtype、含数值域异常的运算或错误隔离图继续使用通用执行器。逻辑 DAG 与公共算子不合并成新的业务黑盒。

一个数组输入串联最多两个 add/subtract/multiply、末尾可选比较或 finite_mask 时，使用预编译寄存器链；常数仍在执行时绑定。中间值留在寄存器，仅根写结果；超出该范围的兼容图继续缓存块执行。保持操作及舍入顺序，不使用 JIT 或乘加收缩。该短链支持连续、负步长、零步长及按步长取数的 SIMD，并合并可连续遍历的相邻轴；没有为 SIMD 打包输入。NEON 将布尔结果批量压缩成连续的 0/1 字节。

具备独占槽位的 typed 根算子直接写结果，后继节点读取该槽。借用输入、重复根和其他需要物化的输出仍明确复制。一般迭代以结果槽和候选缓冲交替写入，保留最后有限状态；候选块的有限性检查与残差归约使用 SIMD。

初态有限，且更新仅由最多两次绝对值不超过 1 的有限系数乘法或加减零组成时，可证明状态始终有限。此时复制初态到独占结果后原地更新，在覆盖前读取旧值计算残差；不会改写输入或上次结果。首次执行完整校验，后续仅在同一迭代作用域复用固定描述，保留每轮预算与逐块取消检查；下一次请求重新检查系数。其余子图保留候选缓冲。Planner 仍保守计入双状态容量，不能把估计值当成实际分配量。普通快照依旧独立，prepared.run 的借用结果规则不变。

执行复制计数与独立 malloc 追踪、RSS 采样共同使用。`tools/heap_probe.cpp` 是仅供 macOS 开发验证的注入式探针，不进入生产 wheel；覆盖指定 malloc 家族调用的申请字节与存活峰值，不把 Python arena 内部分配、mmap 或子进程冒充已覆盖。详见[设计及验收](physical-execution-optimization.md)。

同预算对照使用 `tools/check_execution_scaling.py`，同时测已预热的 NJIT 串行与并行，取该 CPU 预算内更快的路径；不强制短任务用更慢的 NJIT 并行。原有固定单核门禁仍需单独通过。
