# 2026-10-03：按轴原语与有界完整固定带消费者

本记录覆盖构建 `0b0457dc35130e7ac9089c09a8426ce097757954388b4129b532dbb0bec3d6d4`，本机 macOS arm64、CPython 3.12、AppleClang 17、Release、Eigen/Accelerate/Highway 开启。它是一个已执行的成功固定带数学链验收，不表示随机模拟、平台首故障协议或统一引擎整体目标完成。生产源码仍在未提交的共享工作树。

## 1. 实现和数学边界

- 新增 canonical 原语 188–190：带显式初态的按轴 cumulative product/max，以及同一 line 共享排序的多分位数。AST、Typed IR、DAG、Planner、C/F 根、错误与进程运输沿现有契约贯通。
- `compact_path.cpp` 从真实逻辑 DAG 证明完整消费者闭包，保留可编辑节点；没有注册金融模拟黑盒。每批 B 组共享同一资产投影，只保留三份 B×P 状态、一份 P 排序缓冲、资产收益行和贡献状态。B=`min(C,max(1,T/3))`；不同批会重新投影，工作量如实计入计划。
- fan/样本直接写本次独立 C/F 根；14 项统计和合法单组标量后处理执行克隆的原 canonical 节点。贡献按 time→path 左折叠，权重左乘、factor/asset 归约顺序与标量 `std::expm1` 保持。
- 非匹配消费者、公开中间根、跨组后处理、未知几何、scope/isolate/observer 等按原 canonical 路径处理。已准入闭包的数值故障在原节点位置重放，不在错误后切换到未计预算的大图。
- 子图使用独立 Scratch，保留父 captures；小几何释放旧容量。Planner 同时计入 stream、描述符、子图 arena、布局和 Audit 存活；关闭子图融合时预算也关闭相同融合。实际 compact groups/batches/steps 贯通 IPC20。

数学、布局、所有权、取消、预算和完整错误顺序的设计/自审核见 `compact-physical-implementation.md`、`compact-production-static-review.md` 和 `compact-path-static-review.md`。Excel 仍从原逻辑 DAG 生成公式；普通 C++ 不执行导出。完整随机源、旧 seed、coverage、协方差 root/随机变换、limits、JSON 及平台首故障/部分数组协议均排除，不能因本范围通过而关闭 M6/M9。

## 2. 当前正确性和构建证据

| 验证 | 本次终态 |
|---|---|
| Python 全量 | 11,986 通过，含130项私有执行回归 |
| 原生 | 9/9，包含实际 compact 执行体 |
| ASan/UBSan | 9/9；macOS LeakSanitizer关闭，不据此宣称泄漏审计完成 |
| Eigen/Accelerate/Highway 全OFF原生 | 9/9 |
| source / 安装 wheel / extension / worker | 全部身份核对一致，见 identity-compact3.json |
| 新三项 Excel 补充复算 | LibreOffice 25.8.4.2，3夹具/34值通过；Microsoft Excel 新范围仍未重算 |

Python 覆盖实际 C/F/有符号步长、精确 IEEE/错误、两代所有权、prepared 修改后失败/恢复、大→小→大容量、独立结构错误、inline/shared process 与 Hard Stop。prepared process 仍按原接口明确不支持，没有冒充成功验收。历史失败、修复前构建与中间回归放在 `history/`，不改写为最终通过。

CTest 的 LastTest.log、XML 和构建选项保留在 `native/`。最终性能与内存测量各自串行，未与构建/回归重叠；原生回归结束后启动 ASan，ASan 最后数项与下一次 OFF 编译有短暂重叠，不将全部构建/回归过程写成严格串行。新原语 Excel recipe、符号预算和全空根 EMPTY/ERROR 状态测试包含在最终回归；不以 LibreOffice 或旧185/187项 Microsoft Excel记录代替新增三项桌面验收。

## 3. 正式同条件完整时间对照

正式基线从 FIRP 固定 SHA 的源码 AST 抽取四个原始函数和类型声明，保留显式 writable-C 签名、cache/nogil 和 quantile `inline=always`，每函数恰一个 nopython 签名且关闭后续编译。参考生成与 cache 均在 /tmp，不导入或写入平台。数学和编译配置没有为了比较而改写。

每份物理输入只分配一次：NJIT 使用固定签名所需 writable base，C++ 使用共享相同地址/shape/stride的 readonly view；结束后核对输入哈希。CPU=1、同线程预算、完整 fan/14统计/12样本/全部资产贡献、两代独立输出共存、六轮交替首跑；普通入口包括本次绑定/规划、执行和数值结果封装。构图/准备、JIT/tiny/full-size预热及哈希/比较在计时之外，准备耗时单独记录。

| 负载 T/P/F/A/C | ordinary C++ / NJIT | 比值 | 绝对节约 |
|---|---|---:|---:|
| 200/10,000/4/32/1 | 379.224 / 384.198 ms | 0.987 | 4.974 ms，无明显优势 |
| 200/10,000/4/32/8 | 1.781 / 3.074 s | 0.579 | 1.293 s |
| 5/50,000/32/32/32 | 2.554 / 4.044 s | 0.631 | 1.490 s |
| 3/8/4/3/1 | 0.1014 / 0.0123 ms | 8.26 | 增加0.0891ms，仅诊断 |

prepared snapshot 对应约380.714/384.263ms、1.779/3.067s、2.554/4.028s；它保留独立结果，不能用借用缓冲替代 ordinary 结果寿命。原始六对和所有 shape/status/ownership/runtime 路径证据见 `complete-prod-time-ordinary.json` 与 `complete-prod-time-prepared.json`。

`six-pair-analysis.json` 保留每个原始 pair，并补充固定种子的配对及3+3首跑顺序分层 percentile bootstrap、leave-one-out 与观察尾部。只有单机六对，条件区间不代表跨机器可靠性；样本 p95/p99 均退化为六次观测最大值，不是服务尾延迟承诺，不建立新的比例门禁。几毫秒/KB变化只诊断；C1仍需明显完整时间收益。

## 4. fresh 请求分配和 RSS

每 case/backend 的 heap 与 RSS 各用独立 fresh 进程，共18个PID。共同输入、固定签名JIT和 tiny预热在区间外；heap 连续 epoch 含 C++ spec/图/scheduler/plan/prepared、工作区及结果。只 reset peak，不清 live 账本；五次完整 warmup 在 epoch 内；后续两次完整结果替换时保留旧一代。NJIT 四个固定函数的模块/编译状态已在区间外，没有按组合建立 DAG owner；这个接口准备差别保留。

| 负载 | ordinary warm requested heap | NJIT warm requested heap | ordinary / NJIT RSS执行增量 |
|---|---:|---:|---:|
| large C1 | 4,626,508 B | 32,519,400 B | 573,440 / 32,440,320 B |
| large C8 | 9,099,769 B | 48,903,400 B | 3,555,328 / 65,716,224 B |
| short C32 | 18,396,594 B | 8,503,168 B | 10,846,208 / 48,480,256 B |

**短周期完整 owner heap 仍多9,893,426 B。** setup已保留13,735,883 B，warm无结果基线17,171,737 B；较低 RSS 增量不能抵消这个问题，因为 RSS baseline 在 owner构造之后。prepared 短周期 warm heap 为18,342,258 B，结论一致。完整9行cold/warm/retained、RSS、18PID、SHA及源口径复核见 `memory-prod-static-review.md`；全部原始报告保留，没有把差异选择性删掉。

Native worker上界 C1/C8/short为833,070 / 3,084,116 / 5,999,777 B，审计命中1/1/200、8/1/200、32/32/160 groups/batches/steps。stream+descriptor与operator容量有重叠，不能直接相加，也不能把预算与少数audit字段之差当作免费批量余量。完整 owner不全部包含在worker上界中；B2会增加约1.2MB状态，当前先定位owner，不能盲目放宽批量后宣称内存领先。

所有追踪table overflow为0；NJIT有0/13/13次、C++有5/12/12次untracked frees。它们可能释放epoch前合法分配，不是泄漏次数，也不能证明完整allocator覆盖。requested bytes不覆盖全部Python arena/mmap/usable-size；RSS为0.5ms采样，极短峰可能漏采，单fresh观测不代表普遍峰值。报告明确 `complete_allocator_evidence=false`，未关闭完整分配生命周期审计。

## 5. 冻结与复现

`manifest.json` 对归档内完整文件集合、字节数与SHA256核对；`source-snapshot.tar.gz` 是此未提交构建的冻结证据，不参与生产构建，不是并行旧实现。第三方库仍使用仓库原固定archive及许可，哈希见identity；不复制依赖cache、wheel/install树、Numba cache或64MB输入。

`instructions/` 保留运行前说明，里面“尚未运行”是当时状态；实际终态以本README、JSON/XML及原始日志为准。执行驱动的绝对路径可按冻结source安装位置调整，但输入参考AST、source/wheel身份、线程预算、输出范围和结果寿命不能改变。历史lazy-decorator驱动只留作定位，正式结论只来自 `*-production.py`。

当前后续重点：成熟全排序库的完整请求收益、短周期owner/metadata、共享producer；其隔离候选没有被本记录当作已采用。全平台功能覆盖、随机/失败协议、剩余中大型时间和内存差距及新 Microsoft Excel范围仍开放。未提交、推送或发布。
