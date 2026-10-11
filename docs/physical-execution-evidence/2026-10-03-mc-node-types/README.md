# NodeInfo 不可变类型共享验收（2026-10-03）

本阶段把每个 NodeInfo 独占的类型值改为编译期内共享的不可变完整类型。正式候选构建为 `97fdfe5267e55ea83caffc38b333919bcf4c206dc1d37e57eeec500fb86f247c`；[源码、wheel 与两类已安装二进制身份](completed/identity.json)已核对。数学运算、算子目录、CSE key、Program codec、类型推导、worker 值协议与 Excel 公式保持原契约。

**结论：数学回归通过；短周期 requested heap 较上阶段下降 1,405,968 B，仍高于 NJIT 3,911,895 B。RSS 补充观测没有证明相对上阶段的稳定升降，不能宣称整体内存验收闭环。**

## 设计和边界

Builder 按完整类型的 `kind / dtype / axes / shape / semantic_dimension / price_basis / record_tag / fields` 八个字段建立局部哈希表；碰撞仍用 `same_type` 精确判断。发布的节点持有非空 `shared_ptr<const ValueType>`，表随编译器构造阶段结束销毁，节点和图副本各自保活类型。默认未发布 NodeInfo 的 `type()` 保留默认 scalar 值语义。

不存在跨编译全局类型池；数值数据、递推状态与输出结果没有因此共享。group 和 iteration 的类型修订先建立局部可变副本，再冻结发布。Python 返回的类型字典与列表仍独立可编辑；ModelPayload 和 Excel 布局继续按值读取。

相对 [ParentSlots 阶段](../2026-10-03-mc-parent-slots/README.md)，生产改动只有 `compiler.hpp`、`compiler.cpp`、`graph_bindings.cpp`、`native_api_bindings.cpp` 和 `excel.cpp`。公开 C++ 成员类型与结构布局已改变；不能宣称旧 C++ 源接口或 ABI 兼容。Python 属性、结果值协议及公开算子数学接口未改变。详细设计维护在 [MC 轴基础第 8 节](../../mc-axis-primitives.md#8-nodeinfo-不可变类型共享设计2026-10-03)。

独立静态审核覆盖完整字段哈希、精确相等、发布所有权、局部副本修订以及 Python/Result/Excel 边界，未发现确定执行缺陷。静态审核不替代以下运行记录。

## 正确性和完整图等价

| 检查 | 实际记录 |
| --- | --- |
| 完整 Python 套件 | [12,082 通过，无失败或跳过](completed/full-python.xml) |
| ParentSlots 与新增类型共享专项 | [60 通过](completed/focused-python.xml)，其中新增 11 项 |
| 原生 | [9/9](completed/native.xml) |
| ASan/UBSan | [9/9](completed/asan.xml)，macOS 未启用 LeakSanitizer |
| Eigen/Accelerate/Highway 三库 OFF | [9/9](completed/off.xml) |

完整 Python 运行设置 `CALMETRICS_TEST_EXPECT_EIGEN=1`，原先可跳过的 Eigen 检查真实执行。新增专项覆盖 single/thread/process inline/process shared 与 raise/isolate 的八种组合、混合 float/int64/bool 结果、大 int64 key、类型字典外部修改、pickle、图与编译器释放后快照保活、两代独立输出。另三项覆盖不同 dtype/axes/shape。新 Python 夹具先在旧 wheel 上运行 [11 项通过](diagnostic/baseline-public-types.xml)，防止改变已有接口来迎合新测试。

原生测试另外覆盖八字段完整相等、合法 schema 的共享和区分、默认 scalar、独立编译无全局池、所有者生命周期、group/iteration 局部修订与 512 种独立类型。八字段人工值比较不表示每种人工 schema 都是合法编译输入。

固定短周期定义生成 7,405 个节点、28 种完整类型。旧/新全部 metadata、全部 native_nodes 和每个 NodeInfo 类型字典的 SHA256 均为 `94fa298494e245767b2122bec320f08b63512004098877e82a4931e66a0071d0`，逻辑图 fingerprint 均为 `native-3-770eaa22ec3d63dd7e7b8fd608c4ee12`。见 [旧图记录](diagnostic/baseline-public-graph.json)和[新图记录](diagnostic/candidate-public-graph.json)。该逐字段等价证明不自行扩展数值或性能覆盖范围。

## 完整计算链时间

固定已经变换的随机带，计算投影 → NAV → fan → 14 个统计量 → 12 条抽样路径 → 逐资产贡献；CPU 预算 1，同核数、六次交替配对，保留两代独立输出。NJIT 使用冻结的生产函数及其固定签名、cache、nogil 与分位数 inline 语义。普通入口含绑定、输入校验、规划、调度和紧凑结果出口；图构造单列记录并计入内存测量。

| T / paths / factors / assets / combinations | C++ 普通入口中位数 ms | NJIT 中位数 ms | 配对比值中位数 |
| --- | ---: | ---: | ---: |
| 200 / 10,000 / 4 / 32 / 1 | 318.1378545 | 384.4303755 | 0.827480 |
| 200 / 10,000 / 4 / 32 / 8 | 1280.2395835 | 3071.2811665 | 0.416888 |
| 5 / 50,000 / 32 / 32 / 32 | 2157.404875 | 4017.7532915 | 0.536013 |

原始配对数据见 [普通入口](completed/complete-ordinary.json)。8 组合 [F 布局普通入口](completed/complete-f-ordinary.json)为 1282.1514165 / 3068.5620205 ms；[prepared 独立快照](completed/complete-prepared-snapshot.json)为 1278.325 / 3069.1256875 ms。微型负载为 0.1035415 / 0.012146 ms，差 0.0913955 ms，只作诊断。

几 ms 的跨阶段波动不构成本阶段显著提速证据。既有两组大负载约 58% / 46% 的时间收益保持；本阶段减少实际图所有者的重复类型存储。

## Requested heap 与类型多样性负控制

每种负载和实现分别在 fresh PID 测量 requested heap 与 RSS。heap 连续 epoch 覆盖构造、pytest/helper 导入、首次执行、五次预热和两代结果；共同输入与微型固定签名 NJIT 编译位于 epoch 外，两侧规则一致。没有把图或工作区预建到 epoch 外。

| 负载 | ParentSlots C++ warm heap B | 本阶段 C++ warm heap B | 本阶段 NJIT warm heap B |
| --- | ---: | ---: | ---: |
| 1 组合 | 4,429,636 | 4,352,356 | 32,519,400 |
| 8 组合 | 7,948,801 | 7,598,449 | 48,903,400 |
| 短周期 32 组合 | 13,821,031 | 12,415,063 | 8,503,168 |

原始记录：[1 组合 C++](completed/memory-large_c1-ordinary.json)/[NJIT](completed/memory-large_c1-njit.json)，[8 组合 C++](completed/memory-large_c8-ordinary.json)/[NJIT](completed/memory-large_c8-njit.json)，[短周期 C++](completed/memory-short_many-ordinary.json)/[NJIT](completed/memory-short_many-njit.json)。

短周期峰值少 1,405,968 B，仍有 3,911,895 B 的 NJIT 差距。setup retained 从 9,160,160 降到 7,754,192 B。[分阶段构造](completed/owner-staged-short_many.json)的 compiler retained 为 4,253,543 B，较上阶段 5,659,511 B 少同样的 1,405,968 B；pytest 导入的 3,266,115 B 保留在正式 epoch 中。[原构造器对照](completed/owner-original_control-short_many.json)保留，分阶段诊断不替代完整请求测量。

[原生编译拆解](diagnostic/native-retained-types.json)实测同一 7,405 节点定义只持有 28 个类型所有者：编译存活 heap 从 [上阶段](diagnostic/parent-slots-retained.json)的 5,659,463 降到 4,253,495 B。NodeInfo 数组由 2,228,224 降到 1,114,112 B；释放全部共享类型句柄只释放 6,944 B，旧逐节点类型 payload 为 298,800 B。两项之和为 1,405,968 B。该工具拆解后的图不能继续执行，未用于数值、时间门禁；其源码、静态库、工具链与二进制身份见 [native-probe-identity.json](diagnostic/native-probe-identity.json)。

共享也有成本：[4,096 种独立合法 schema 的旧构造](diagnostic/unique-4096-baseline.json) / [新构造](diagnostic/unique-4096-current.json)显示 retained 从 23,876,664 增到 24,040,504 B（+163,840 B），peak 从 25,486,864 增到 25,865,912 B（+379,048 B，约 1.5%）。调用方 schema 同在 epoch 外，compiler 与 graph 都在 epoch 内构造并保活。单次时间 +1.394 ms 仅作诊断；此负控制不运行数值计算、不与 NJIT 比较，也不证明所有图都节省内存。

heap 探针无 table overflow，但存在 pre-epoch/nonintercepted frees，未证明全部分配器覆盖；CPython arena、mmap 和非拦截分配器仍有盲区。

## RSS 原始差异与相邻复测

原正式 RSS baseline 位于 owner 构造后、首次完整执行前。1 组合 C++ / NJIT 执行增量为 262,144 / 32,538,624 B；8 组合为 3,506,176 / 65,699,840 B；短周期为 17,793,024 / 56,098,816 B。短周期旧 baseline/peak 为 207,503,360 / 217,333,760 B，新为 203,866,112 / 221,659,136 B，原单次绝对峰值增加 4,325,376 B，保留这个观测。

为检查 MB 级差异，另做 [8 个相邻 fresh PID 补充观测](diagnostic/rss-owner-pair-summary.json)：3 对普通 native 旧/新和 2 次 NJIT。共同 64,016,640 B 输入、运行时与微型预热仍在 epoch 外；RSS baseline 前移到 owner 构造前，构造、第一次完整执行及两代结果替换均在同一采样 epoch 内。旧源码由验证过的 ParentSlots 快照解包，旧/新分别核对 87 个源文件与两类已安装二进制，禁止把新源码身份套用旧 wheel。六项 native 记录均事后与冻结 NJIT 逐项对照；八项记录均检查输入不变和两代输出所有权，NJIT 单项不重复跨后端等价验收。

| 三对相邻 native 复测 | 新减旧绝对 peak B | 新减旧 whole-owner delta B |
| --- | ---: | ---: |
| 旧 → 新 | +3,227,648 | +2,064,384 |
| 新 → 旧 | +737,280 | +1,359,872 |
| 旧 → 新 | −6,258,688 | −4,456,448 |

旧 native peak 为 219,496,448–227,229,696 B，新为 220,971,008–224,935,936 B。两侧观测范围重叠，配对差异双向变化，未证明稳定的 RSS 增长或降低。NJIT 两次 peak 为 242,581,504 / 247,627,776 B；这些范围内 native 低于 NJIT，但三对观测不能证明全负载或全机器的 resident-memory 优势。两侧执行 requested-allocation 增量相同，构造改变分配器驻留和空闲页复用是一种待验证解释，不是已证实原因。

补充 driver 与 [串行入口](drivers/run-rss-owner-pairs.py)均保留，原正式 driver 和原记录未重写。RSS 监控目标间隔为 0.5 ms，未保存实际采样间隔；idle 增量为零且不从结果扣除。补充只观察两次完整执行，未覆盖五次完整预热与 owner 释放后驻留；没有 worker/PSS/共享页聚合。RSS delta 是进程驻留页观测，不能当工作区，也不能用它关闭 requested-heap 差距。

## 完整性与尚未覆盖的范围

- [manifest.json](manifest.json)保存全部归档文件的字节数与 SHA256。运行 `python verify_manifest.py`只检验归档字节完整性，不执行数学或性能验收。
- 源码快照（本机保留的历史归档：`source-node-types-final.tar.gz`，未纳入此次提交）及[清单](source-node-types-final.tar.gz.json)保存 241 个源码、测试、工具及构建输入；另含快照清单与构建身份。第三方归档仅记录版本和哈希，外部平台树、二进制、非文本夹具未全打包，不能宣称这是独立可构建的完整 sdist。
- `completed/` 保存构建、完整/专项回归、性能及构造诊断；`drivers/` 保存实际脚本、冻结 NJIT 与 consumer；`diagnostic/` 保存新旧完整图、独立类型负控制、原生拆解与 RSS 补充原始记录。原始绝对路径是实际运行位置，迁移复测须提供相同工具链和输入，不能把历史代码作为生产 fallback。
- [12 项终态](completed/acceptance-status.json)全部 exit 0，另有三类 [原生终态](completed/native-regression.json)和八项 RSS 补充终态。exit 0 或 manifest 校验不自行扩大验收覆盖范围。

未关闭完整随机流、覆盖率、协方差根、因子变换、limits、JSON 封装与按阶段首错前缀的业务请求验收；也未证明所有平台算法、CPU/布局或 Microsoft Excel 187–190 重算通过。下一步应扩大真实业务计算链路、寻找秒级耗时和大额内存瓶颈；短周期 heap 与稳定 RSS 优势仍需分别验收。未提交或推送。
