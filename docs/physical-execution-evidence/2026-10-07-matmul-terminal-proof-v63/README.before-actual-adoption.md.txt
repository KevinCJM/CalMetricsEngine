# V63：矩阵链资格完成，跨布局修复候选复验

2026-10-07。实际源码仍为V61；C和D都只在隔离目录，未提交、推送或部署，整体目标未完成。

C完整正常Python16370、ASan/UBSan关联2792、OFF 2400通过/392库关闭资格skip；三种构建native各9/9。ASan未开启泄漏检查，不能称为泄漏验收。C最终48个fresh进程（四组轮换、同CPU1、owned输入准备、原顺序算术、两代独立C/F根和全输出核验）全部完成。相对V61约6.3→0.64–0.66秒，申请峰减少512MiB；但优化NJIT约0.57–0.59秒且申请峰基本相同，不宣称优于全部NJIT或目标完成。

C另外24个fresh跨布局完整请求揭示F输入/C结果约0.136→0.270秒退化。新早选绕过既有Eigen column/tile，是属实的问题。D抽取共用的同序连续内存资格选择器，资格不符继续已有列/分块库路线；未增加算子、数值缓存、参数、IPC字段或线程。依赖版本不变，调用者要求的C/F结果保留。数学两步顺序、NaNInf、零符号、错误和独立所有权不变，Excel逻辑契约无需变更。详见[设计与官方来源](design-layout-guard.md)。

D build_id `4c39b63bd64febb6a076deacddd620f389f09206324ba6395a7b789d3c43aa62`；93份source/config和182份测试源码冻结。D针对性Python230（新增144）与native9/9通过，原生另覆盖24个跨布局除法失败前缀类。36个fresh跨布局和60个fresh矩阵链完整请求已通过，全输出/旧结果/非别名核验保留。D完整正常回归现已完成：Python 16514 项及 native 9/9 通过，无失败、错误或 skip；终态日志见[正常资格](qualification-layout-guard-normal.json)。SAN 关联2936项与 native9/9现已通过，无失败、错误或 skip（未启用泄漏检测），见[终态SAN资格](qualification-layout-guard-asan.json)。OFF 已完成：2544通过、392按库资格skip、native9/9通过，无失败或错误，见[终态OFF资格](qualification-layout-guard-off.json)。正常/SAN/OFF全部资格已通过，实际接入待执行；当前进度见[阶段记录](qualification-layout-guard-progress.json)。实际源码仍 V61，不能把正常回归通过视为全部资格或实际接入。

最终数据、每组中位数和峰值见[候选验收](acceptance-qualified-candidates.json)。原A48/B18、C48与各原始日志不删；最初取消/bad_alloc失败和修复依据见[早期自审核](self-review.md)，其历史待验项以本记录为准。最初正常里程碑保留在[快照](README.normal-milestone.md.txt)和[原里程碑JSON](acceptance-normal-milestone.json)。`in-progress/`只表示该UTC时刻的运行日志，不是完整验收。实验source/driver为不可执行.txt，不进入当前构建或CodeGraph代码解析；wheel和机器码不复制入仓库。

仍需D完整资格、适用时的实际同源接入，以及其他主要平台链和全面分配/复制审计。几KB/ms变化只作诊断，未改阈值；不是因微小尾差或内存差阻断采用。未做跨平台或live Microsoft Excel运行验收。

2026-10-07按用户要求清理212个闲置CMake构建目录，释放21.8GiB，可用空间约2.9→25GiB。653项源码/未提交文件/实际运行二进制哈希不变，实验源码、wheel/installed及全部验收日志保留；仅保留一份继续开发的增量构建目录。历史identity中的已删除build路径须重新构建才能再跑ctest，不影响保存的终态日志或现存wheel/worker身份。详见[清理记录](build-cache-cleanup-record.json)。未提交、推送或部署，实际计算源码仍V61。
