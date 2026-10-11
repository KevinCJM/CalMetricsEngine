# 固定 transformed tape 投影测试与测量准备

2026-10-03。仅生成源码，未运行数值、编译、NJIT 预热、性能或内存测量。

依据当前 AGENTS、计算纪律 D1–D8、Excel 纪律 X1–X10，以及 `/private/tmp/calmetrics-platform-mc-next.md`。CodeGraph 先定位 FIRP `monte_carlo_projection_kernel`，随后冻结其 intercept-first、coefficient-first/factor 升序、weight-first/asset 升序、step→path 贡献累加和最终除 path_count 的数学路径。CME 不反向 import FIRP。

`reference.py::frozen_one` 保留原数值循环及首故障元组用于参考；正式对照限定健康输入的完整 period 和贡献输出。`service_chain` 对每组合重算该冻结内核；`shared_chain` 是额外的 NJIT 共享 comparator，以 A 长度临时向量复用 affine/expm1，不形成 N×A 全量 L/R。

AST 图共享同一 affine/expm1 依赖，组合权重显式广播后先乘再按资产 sum_axis；period 经 swap_axes 返回原 [path,step]。每资产贡献使用同一权重输入的 first(slice)、scalar-first multiply、原 sum 顺序和最后除 P。多组合不能拆成独立请求重复公共节点。默认未选择 L/R 中间根，完整 period 与所有贡献根均返回；此输出合同不同于平台最后只返回 fan/sample 的完整 MC 请求。

`tests/test_mc_projection_primitives.py` 只新增这一测试文件，不修改共享测试/生产实现/权威文档。覆盖原语、AST/Typed ordinary/prepared、布局、readonly、独立输出、边界/IEEE、几何/axes、选根和退准入情形。新 direct out 继续 C-only；graph 可指定 F，测试实际逻辑值和连续布局，不能只检查 order 标签。Excel recipes/cases 由主任务同步；这些原语测试没有替代真实 Excel 重算。

`bench.py` 可通过 `--install` 或 CME_INSTALL 选择构建，通过 `--implementation` 选择计时侧；无缺包静默源码/数值回退。每个正式运行先完整差分，再交替配对计时，普通入口含绑定/计划/封装，prepared 用独立 run_snapshot。只保留每侧一个旧结果，调用过程中生成独立新结果。输入、图构造和明确 JIT 编译成本分别记录；不存在保存初始 parity 数组导致第三份结果常驻的问题。审计记录真实 build ID、binary/reference/driver SHA、完整 schema、native copy/scratch/Planner 字段。

默认 CPU=1；`--cpu` 可调整 Scheduler/库预算，但 NJIT 无 prange，所以 CPU>1 明确标为探索，不能冒充同有效核数结论。对等的主要时间/内存验收用 CPU=1。计时与 heap/RSS 分开；主任务串行使用已有 benchmark_memory helpers，不在这里复制内存框架。

主任务先运行小规模 correctness，确认没有全量 L/R 巨额 workspace 后才能跑大形状：

```sh
PYTHON /private/tmp/calmetrics-mc-projection-integration/bench.py --install INSTALL --n 120 --steps 10 --factors 4 --assets 3 --combinations 2 --correctness-only
PYTHON /private/tmp/calmetrics-mc-projection-integration/bench.py --install INSTALL --n 2000000 --steps 200 --factors 4 --assets 32 --combinations 1 --reference service --output OUTPUT.json
```

同一个 n=2M/factors=4/asset=32 case 依次 C=1/8/32，service/shared 均对照；另 n=250k/factors=32/asset=32，C=1/8 按资源预算串行执行。两种代表形状都恰好不超过 8M transformed-factor 元素。完整两结果 period 容量为 `2*8*C*N`，N=2M/C=32 即 1,024,000,000 字节；除此还计输入、线程 arena、tile、元数据和库工作区，不能把 output 留存当零成本。C=32 长图也须核对 compile 节点预算，超限明确失败，不截断根。

尚未完成：随机 primitive tape/旧 seed、相关矩阵根算法、完整 first-fault prefix 协议、NAV/peak/drawdown、fan/tail、available 的业务选择/回填及平台 HTTP 集成。健康 fixed-tape 数学链通过不关闭这些缺口。

## 私有分块闭包回归准备

追加的测试明确区分逻辑图与物理执行：raise/typed 的共享 affine→expm1 精确闭包用 native_chunks 审计检查无 numeric arena、独立根直接写入及跨块贡献左折叠；isolate、公开 returns 中间根或额外独立算子根保持现有 canonical 路径。C/F/strided/negative 输入按逻辑坐标验收，不能把非 C 输入无条件称作退准入。F 根检查真实连续布局和值，不以标签代替写入正确性。

跨三块有限多根对照使用同 native canonical 的 isolate 路径，以分类和有限位模式精确比较，独立数学参考及 NJIT 仍由主任务另行验收。极端 IEEE 用同一 strict policy、额外公开 R 强制 canonical 退准入，逐 selected 根比较值与 status；isolate 对非有限 scalar 的既有 status 4 则单独验收，不能把两种 policy 的差异误报为融合差异。空 T/P 时 period 保留空几何；空 scalar contribution 按既有 sum(empty) 失败；空 factor 返回初态，空 asset 的 period 返回 0。极端 IEEE 不跳过 0*Inf，不声称 NaN payload 完全一致或完整 MC 首故障协议已实现。

强制 thread、process-inline、process-shared 通过同一 Scheduler 的现有配置及两段区间触发，检查全部 period/贡献、CPU token、实际传输和关闭后的独立结果寿命。Planner 测试验证有界 tile 的 scratch 不随 outer size 增长、完整 input/result 总预算拒绝与准入；它仅证明 Planner/执行审计一致，不能当 heap/RSS 实测。same-shape 的 coefficients、weights 和 scalar divisor 更新必须重新读取，独立旧结果保持有效。

未知 iterate seed 的 DIVIDE_BY_ZERO 回归覆盖 affine 后继续 sum_axis/stack 的多跳保形 body，避免 geometry 占位被当真 rank；已知独立坏 coefficients/initial/weights 则继续结构拒绝。z/divisor 均通过正式 parameters 入参。以上新增用例只生成源码，未由本代理运行；主任务统一串行执行。

扫描合并候选仅在 contribution 的 weights 节点与某个 period 的 weights 节点相同时复用该 period 逐资产 product，选择一个 carrier；无匹配消费者继续独立扫描。对应回归覆盖重复 period/贡献根、部分选根、相同内容但不同权重绑定的后续更新、同 sum 的不同常量/参数分母及逆序 period 展示顺序。逐根对照同一 strict policy 的 canonical 原表达式，保留状态与有限位模式检查，不能通过重复 carrier 给贡献二次累加或按值合并不同输入身份。
