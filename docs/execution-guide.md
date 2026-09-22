# 执行与性能指南

[文档导航](README.md) · [使用手册](user-guide.md) · [算子参考](operator-reference.md)

本页说明当前 Planner 的行为，不承诺任意数据上并行都更快。依据是 [planner.cpp](../cpp/planner.cpp)、[默认配置](../cpp/include/calmetrics_engine/planner.hpp)、[原生运行时](../cpp/native_runtime.cpp) 与 [async 适配](../src/calmetrics_engine/runtime.py)。

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

## 3. Planner 的选择顺序

这里的 rows 是**请求的区间条数**，不是观察点数 T；work 是 lowering/CSE/融合后物理 DAG 的结构性成本估计，不是 FLOPs、毫秒或节点数。

1. 校验输入、区间、dtype/轴、图身份，估算输入/输出/状态及 scratch 容量。
2. 若 hard_stop=True，选择 process。否则同时满足 `rows >= max(2,min_rows_per_worker)`，以及工作量达到 process_work_units **或**输入字节数达到 process_input_threshold_bytes，才走普通 process。
3. 未选择 process 时，若为标量根图、`cpu_budget>1`、`0<rows<cpu_budget`，存在至少两个足够重的独立编译分支且总成本达到 dag_branch_work_units，则考虑 thread 的 dag_branch 计划。共享上游与融合组先合并，不按每个节点开线程。
4. 否则，当 `work>=thread_work_units`、`rows>=2` 且 `cpu_budget>1`，选择普通 thread。
5. 其余为 single，在调用者的原生线程计算。
6. 内存预算可降低 worker 数量；降至 1 仍超预算则失败，不隐式修改算法或精度。当前 CPU 准入还可能使请求排队；计划 worker 数不等于任意时刻真正并发数。

普通进程数初值为 `min(cpu_budget,max_processes,max(1,rows/min_rows_per_worker))`（整数除法）；普通线程数初值为 `min(cpu_budget,rows)`。Hard Stop 即使仅一个区间、一个 CPU 也需要进程隔离。单区间普通时序根不自动获得跨时间递推并行。

分块优先考虑 product_ids 提供的足够产品组，否则按区间；使用物理成本加权，而不是简单按行均分。DAG 分支是区间并行不足时的受限补充。目前仅 scalar 根适用该分支策略。

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

**自动共享运输条件**：已经选择 process，且满足 hard_stop、输入已是 SharedInputBundle、输入字节数达到共享阈值三者之一。单独超过 8 MiB 不会强制多进程，也不保证普通单线程请求新建共享内存。

小型 process 请求可使用显式计量的 inline IPC。大输入或 Hard Stop 路径由父进程持有共享映射、worker 只读附加；worker 将不同结果块写入不重叠区域。共享输出返回的 NumPy 视图持有原生 owner，无最终整块结果复制。

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

prepared 当前只接受 single-lane 计划，是稳定输入/几何的重复执行入口，不是并行 execute 的通用替代或流式状态容器。需留存结果用 run_snapshot；借用输出会被覆盖，参见[使用手册](user-guide.md#9-重复执行与结果所有权)。

## 7. 如何判断本次执行是否符合预期

- plan.metadata()：lane、reason_codes、parallel_dimension、chunks/branch_tasks、预计 worker/内存和 simd_nodes。
- result.audit：实际 cpu_tokens、native_threads/native_processes、排队/计算耗时、复制字节数、共享输出所有权及 native_chunks。
- 当前图的 native_chunks 报告 arena、复制和融合等数据，**不提供逐节点 ISA/向量计数**。直接算子 audit 的 isa/vector_elements 可诊断该次单算子执行，不能据此宣称整图每个候选节点都实际向量化。
- engine_build_id、plan_fingerprint、registry/IR 版本和 input_dtypes/output_dtype 用于追溯。凭据不是外部认证协议。
- estimated_total_memory_bytes 是输入/输出/运输/arena/scratch 估计，不是 OS RSS 上限；峰值 RSS 另行测量。

性能排查顺序：先验证公式与区间 → 看复制和布局 → 看共享依赖是否复用 → 看计划分块/排队 → 最后调整阈值与 CPU 预算。重复固定任务考虑 prepared；需要保留结果时比较 run_snapshot，不能拿复用裸缓冲的耗时冒充独立结果成本。

完整性能门禁见 [AGENTS.md 第 16 节](../AGENTS.md#16-performance-evidence) 和 [门禁工具](../tools/check_phase2_performance.py)。本页示例验证功能路径，不提供性能提升或跨平台运行证明。
