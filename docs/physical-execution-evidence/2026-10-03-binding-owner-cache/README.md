# 普通输入缓存：执行保活与完整 owned-input 验收

状态：本地开发阶段，未提交或推送。2026-10-04 完成当前构建复验，整体性能目标仍未完成。目录日期保留本阶段起始日期；前阶段测量与最终构建分开存放。

## 1. 设计与数学边界

普通 `AdaptiveScheduler.execute` 原来隐式缓存强输入 pin，调用结束后仍保留旧 raw/chi；下一次生成输入时，两批大数组同时存活。缓存布局元数据具有复用价值，但缓存输入数值没有必要。显式 Prepared、ModelPayload 和 shared input 的所有权继续遵守其原契约。

完整设计见[按轴图第 11 节](../../mc-axis-primitives.md#11-普通输入缓存与执行所有权)。本轮只修改 `cpp/native_api_bindings.cpp` 的所有权管理，算子仍为 190 个；不新增算子、不改变数学、DAG、Planner、worker 值协议或 Excel 公式。

- 只共享已经证明兼容的绑定描述及计划。当前请求内容每次读取，不能因地址/shape 相同复用旧数值。
- mapping 首次绑定使用已校验 pin 的 owner；缓存匹配一次取值并先强持有，再检查完整几何。执行与保活必须使用同一个对象。
- 普通缓存只保存不可解引用的 identity、地址和布局；每次重新验证当前对象及底层分配。地址重用可以通过完整当前校验，但不能跳过校验或读取失效对象。
- 本次执行的强引用跨 GIL release、任务完成、结果构造和异常处理，Python 对象仍只由持有 GIL 的调用线程释放。匹配时局部持有原 BoundData，防止重入替换缓存使当前绑定失效。
- 放弃普通 owner 时重建无 metadata 的已校验 dtype 描述，避免 dtype metadata 间接持有大数组；不修改调用方数组或 metadata，不改变 dtype/字节序校验。缓存命中后不重复重建描述。
- 当前输入、递推状态、必要 scratch、常驻图/计划和两代独立结果仍计入成本。这里只移除调用结束后不必要的输入及 metadata 引用。

这是物理所有权变化。数学节点及 Excel 公式的依赖和含义不变；本轮不启动 Microsoft Excel 重算，也不关闭现有 187–190 的真实重算欠项。

## 2. 自审核与负控制

[生命周期测试](../../../tests/test_binding_cache_lifetime.py)覆盖原绑定的回收、Prepared 保活和 resize 守卫、动态 mapping/dict 子类、base 校验回调、当前内容及两代结果独立。

无 weakref 的普通缓存避免新增 NumPy resize 限制。早期 weakref 候选因此未采用，原失败 XML/日志及源码另存 `negative/`；其中混合了已修正的夹具问题，不能把全部失败都归因于 weakref。另复现过“校验数组 A 后再次从 mapping 取得 B 保活”的不安全候选，改为对实际校验对象持有本次引用；这些候选不进入最终生产路径。

dtype metadata 的八项负控制，每项由极小支持数组间接引用一个 8 MB 数组，覆盖 float64/int64/bool/uint8 输入及 starts/ends/product_ids/parameters。旧 a54 构建八项均因 metadata owner 未回收而失败；最初 uint8 夹具含 2–7，与既有 0/1 掩码约束冲突，修正为合法 0/1 后仍复现同一引用问题。原失败 XML、夹具和更正后失败保留在 `negative/`，没有修改掩码契约。

当前构建 `046eaabdc52802925698bc6cc6aead90a04e1e147391ca5aec57b3db64b57e99` 的生命周期 37 项已通过。全量 Python 12,304 项、绑定专项 363 项以及真实扩展 ASan/UBSan 363 项均通过，无 skip；原生、ASan/UBSan、三库 OFF 各 9/9。专项包含在全量中，不重复相加。macOS 未启用 LeakSanitizer。Python 回调测试是确定性的边界证据，不宣称实现跨 GIL worker 屏障或完整 allocator 审计。

`preceding-leased/` 保留 a54 前阶段完整回归、性能及原沙箱 CTest 7/9 结果。两项失败为 shm_open 权限；后续实际结果独立记录，不能删掉原失败。该目录中的 test source hash 捕获早于最后一次夹具调整，只是历史记录，不能用作最终测试身份。

## 3. 公平对照范围

双方每次调用同一冻结平台 fixed-signature NJIT 输入准备，计入实际 seeded RNG 分配及耗时；随后完成 normal/Student 变换、资产投影、NAV、fan、14 项统计、至多 12 条样本及全部资产贡献。没有额外 readonly replay copy，未加载或调用实验原生 RNG。

CPU=1，原生库线程=1，NJIT 完成预热并禁用新增签名编译。全部所选具名根比较 shape/dtype/status，浮点 rtol=2e-12、atol=2e-13，概率等离散结果精确比较。保留两代结果，并验证旧结果不变及输入只读。

ordinary 计时包含绑定、校验、计划、执行和结果封装；静态模型 fixture 与构图在计时外，纳入连续内存 epoch。协方差根生成、coverage、limits、JSON、平台按阶段首错仍排除；本范围不等于完整业务请求迁移。

requested heap 使用连续 setup/cold/warm epoch，保留 setup 存活分配及两代结果，不将各阶段峰值相加。fresh-process RSS 另行采样；Python arena、mmap、非拦截 allocator、stack 及既存 Numba TLS 等盲区明确保留。

## 4. 最终结果

| 负载 / 分布 | C++ / NJIT 时间 ms | C++ / NJIT warm requested heap MB |
| --- | ---: | ---: |
| T200/P10000/F4/A32/C1 normal | 477.255 / 526.896 | 65.106 / 96.521 |
| T200/P10000/F4/A32/C1 student | 605.600 / 649.548 | 97.118 / 112.521 |
| T200/P10000/F4/A32/C8 normal | 1438.150 / 3213.123 | 68.432 / 112.907 |
| T200/P10000/F4/A32/C8 student | 1571.948 / 3339.635 | 100.522 / 128.907 |
| T5/P50000/F32/A32/C32 normal | 982.408 / 4205.262 | 110.762 / 72.528 |
| T5/P50000/F32/A32/C32 student | 1002.656 / 4212.948 | 115.090 / 74.528 |

31 个性能进程任务均正常结束：7 个计时对照（每个六轮交替）、12 个连续 heap 观测，以及 Student C8/短宽 C32 每方三次 fresh PID 的 12 个 RSS 观测。MB 为十进制；计时与内存测量分别运行，没有并行数值任务。

C8 时间减少 53–55%，warm requested heap 减少 28–44 MB；Student C8 的三次 RSS 增量中位数为 **102.826 / 145.736 MB**，少约 42.910 MB。普通缓存释放与原 85 构建使用同一 RNG 准备对照，Student C8/短宽 C32 的申请峰分别减少 **79.480 / 64.841 MB**，不是额外 replay copy 产生的收益。

短宽 C32 时间减少约 76%，但 normal/Student 的申请峰仍分别比 NJIT 多 **38.234 / 40.562 MB**；Student RSS 增量三次中位数 **142.705 / 128.139 MB**，也没有优势。C1 时间只减少约 7–9%，仍需实质提速。微型请求多约 **0.148 ms**，仅作诊断，不作为门禁。RSS 原始范围、绝对峰值、baseline 和 idle 控制在各报告中保存，不以较好的一次测量隐藏其他结果。

完整源码与 wheel/worker 身份一致；87 个生产文件与原 85 阶段仅绑定文件不同。`source-production.tar.gz` 保存这 87 个文件，`source-tests-tools.tar.gz` 保存实际测试、接口、工具和第三方分发源。Python 回归记录含测试源 SHA；最终 XML 的 12,304 个用例身份不同，零失败/跳过。

**整体目标未达成**：短宽 C32 内存、单组合提速、其他平台重型链路的当前完整请求验收仍开放。协方差根、coverage/limits/JSON 与阶段首错没有被纳入这一成功链；不能由本轮数值结果推断它们已经迁移。下一步优先评估现有状态批次的时间/内存取舍及现有成熟向量数学库的有界 producer 使用，先筛选，保持数学、离散和所有权契约；不新增功能性算子。


## 5. 证据与复现边界

`drivers/` 保留实际外部对照脚本及冻结参考，属于测试证据，不是引擎生产依赖。脚本包含原始绝对路径；复现时须恢复所引用的固定目录或逐项调整路径，并重新核对参考/模板/源码 SHA。它们没有被接入生产运行或默认测试收集。

本机为 macOS arm64 / CPython 3.12；其他平台和业务服务迁移不由本轮测试认证。

源码/包构建身份、实际命令、原始 XML/日志和完整性能输出已归档到 `final/`；文件 SHA 见 `manifest.json`，使用 `python3 verify_integrity.py` 复核完整性。该校验只证明归档未变，不重新执行数学或性能验收。完整 wheel 与 worker 不在源码仓库中重复保存，仅保存构建和二进制身份；本机安装目录在对应 identity 中记录。
