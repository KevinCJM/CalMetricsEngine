# 短周期大状态批次：正式复验完成，整体目标未闭环

2026-10-04，目标与范围见[设计第12节](../../mc-axis-primitives.md#12-短周期大状态批次2026-10-04)。仅优化已有算子组合的物理执行；没有新算子、数学口径、布局/API或Excel公式变化。生产 C++、wheel 与 standalone worker 构建身份为 **7abae5b212f577f4ab68b18fb5af0ed177c091b938e70dbae0a77b2ead832c0a**；ASan扩展身份为 **25105accf9e288a084aa650b1e3f4b5a5be96c196f40b7113bedcc7dd690ff5a**，两者87生产源相同，编译参数不同。

正式包全量 **12,319通过、无skip**，专项296包含在全量中；实际ASan/UBSan扩展 **52通过、无skip**，原生、ASan/UBSan、禁用三库分别 **9/9**。正式包31个串行性能任务均正常结束，逐文件/source/test SHA与wheel/worker身份已核对。`progress.json`只记录状态，结果证据是XML、日志与原始性能JSON。整体目标继续开放，未提交、推送。

## 设计与自审核

只读 raw/rotation/means/scale/资产参数共享，四组的 NAV/peak/drawdown 和贡献独占；每组所有fold及排序公式不变。原准入条件保留；只有T<C且完整组合状态至少16 MiB时限到四组。Student scale和系数pack只计算一次，实际raw及asset工作量按批次数计入Planner。post同步返回后才能复用下一批状态；child Scratch不借用状态作为可写arena，根写入本次独立结果。

正式296项包括新增15项large-state测试，全部选定根逐位对照公共canonical图，并覆盖C/F、单线程、线程、共享进程、两区间、只读、两代独立结果、错误类型/消息和错误后恢复。native几何预算测试覆盖16 MiB上下界及未知payload，不能把它当作真实大状态执行的替代。旧小状态/负stride/nonfinite/Prepared/拒绝案例保留。现有logical DAG/Excel展开不变；本轮没有新Microsoft Excel重算，也未关闭187–190的旧欠项。

## 隔离候选：已结束的数据

`screens/cap4`（c0ba665e）与 `screens/cap2`（28de24c6）从046冻结源出发，只改compact_path.cpp。它们没有正式版的大状态分派条件，因此不能当作正式版完整验收。

CPU1、same frozen warmed NJIT RNG prep、普通入口、全部fan/14统计/样本/贡献和两代结果：四组normal/Student为 **1.497/1.510秒**，NI **4.202/4.212秒**；requested heap为 **77.155/81.483 MB**。两组为 **2.102/2.118秒**，heap **74.754/79.082 MB**；只多省2.401 MB却多耗约0.6秒，未采用。

四组Student RSS三个独立PID增量为 **112.067/108.298/108.184 MB**，NI **133.972/133.677/140.050 MB**，中位数低25.674 MB（约19%）。这些RSS记录不能关闭requested heap剩余差距，也不能代替正式包复证。旧046及两组控制的原始数据保留，没有调小输出、放宽容差或增加核数。

计时包含每次RNG分配、绑定/校验/规划/执行/结果返回；构图和静态fixture在计时外、在连续heap/RSS epoch内。双方不承担额外readonly replay copy；原参考会原地变换自己的新带，引擎保持自己的输入只读。实验native RNG没有被加载。排除协方差root构造、coverage、limits、JSON和平台阶段first-fault，不能声称整个平台请求完成。

requested heap为连续追踪申请峰，未覆盖Python arena内部、mmap、非拦截分配器、栈和既有Numba TLS；未跟踪释放保留。RSS包括allocator驻留及0.5ms采样盲区；不是完整分配/复制审计。所有数据范围都以驱动和原始JSON为准，不能相加阶段峰值或把累计分配当峰值。

## 正式包完整链复验

CPU1、ordinary、六轮交替配对；相同固定签名NJIT RNG准备在双方每次调用内，全部fan/14统计/样本/贡献输出、状态、只读及两代独立结果通过对照。MB为十进制。

| 负载 / 分布 | C++ / NJIT 时间 ms | C++ / NJIT warm requested heap MB |
| --- | ---: | ---: |
| T200/P10000/F4/A32/C1 normal | 478.765 / 527.682 | 65.106 / 96.521 |
| 同上 Student | 605.280 / 649.558 | 97.118 / 112.521 |
| T200/P10000/F4/A32/C8 normal | 1436.881 / 3222.770 | 68.432 / 112.907 |
| 同上 Student | 1569.692 / 3337.846 | 100.522 / 128.907 |
| T5/P50000/F32/A32/C32 normal | 1495.599 / 4195.211 | 77.155 / 72.528 |
| 同上 Student | 1510.749 / 4210.606 | 81.483 / 74.528 |

短宽C32较046的连续热申请峰减少 **33,607,184B**，代价是重复producer令耗时从约1.0秒到约1.5秒；当前仍比NJIT快约64%。C8无新增实质波动：normal/Student比NJIT快约55%/53%，申请峰少44.474/28.385 MB。C1只有约7–9%提速，仍待改善；tiny多约0.151ms只作诊断。

Student fresh PID RSS三次：C8 C++为103.350/103.105/103.334 MB，NJIT为145.703/146.260/146.571 MB，中位数少 **42.926 MB（约29%）**；短宽C32 C++为107.790/105.726/108.249 MB，NJIT为130.138/133.186/135.512 MB，中位数少 **25.395 MB（约19%）**。短宽requested heap仍多4.627/6.955 MB，单列欠项，不能用RSS优势掩盖。申请峰含不同allocator盲区，RSS采样也非完整分配审计。

全部原始31个JSON/log、终态PID/命令与摘要在 `final/fair/`、`final/supervisor-status.json`、`final/acceptance-summary-final.json`。复证只覆盖下述完整成功数学链及输入准备；不声称平台全请求、其他主负载或整体目标完成。

## 来源与复现

- `final/`：正式87源、wheel/worker身份，full/focused XML/log、真实ASan扩展52项、三类native日志/配置，以及全部31个正式性能JSON/log与汇总。
- `screens/`：两个候选源差异、身份、全部raw timing/heap/RSS和运行状态；均为终态。
- `drivers/`：构建、串行运行、已确认PID的监督和相同冻结NJIT准备/参考；`baseline-support/`保存上阶段未变的依赖驱动。

完整重建源先解包 `final/source-tests-tools-baseline.tar.gz`，再解包 `final/source-production.tar.gz` 和 `final/test-delta.tar.gz`，以manifest中的逐文件SHA核对。临时目录、安装路径和driver导入路径是原执行路径；换位置需调整受控路径并重新冻结身份，不能把源与旧wheel混用。sanitizer实际扩展要求加载本记录的Apple runtime，macOS LeakSanitizer不受支持，不宣称泄漏检测通过。

继续项：C1提速幅度、短宽requested heap、其他矩阵/模型/平台主负载、完整分配审计及平台业务请求边界保持开放。正式回归和本链复证已归档，不能据此关闭整体目标。
