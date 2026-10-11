# V55：GMM 有界后验重算筛选，尚未采用

本轮只改隔离副本。实际引擎仍为 V54 / `7f3d170e3cad9e93d6d454e435e65a9e287db425a3a9e3ba4fb01f5f6bdd9306`；315个原有源码、测试、工具、规范文件在归档前保持原hash。没有提交、推送或上游项目修改。

## 做了什么

先用当前 CodeGraph 的显式文件/行范围核对 GMM 与 Planner，确认原生只保留一块 NK 后验。调研 [NVIDIA cuVS 官方分块方法](https://docs.nvidia.com/cuvs/user-guide/api-guides/clustering-guide/gaussian-mixture-model)、[Armadillo 原作者的 MapReduce EM 方法](https://arxiv.org/abs/1707.09094)及[公开 API](https://arma.sourceforge.net/docs.html#gmm_diag)。cuVS GPU 包不用于本机 M1 CPU；Armadillo整模型训练不能未经证明替换现有初始化、停止、floor和错误契约。

隔离候选复用现有非拥有 Eigen 密度/加权归约，128行后验、旧均值、持续按样本顺序归约，重算旧E步以保持中心化方差。第二版以有界F输入tile复用原有行SIMD；最终C版补回均值/方差失败时整分量原始和的写入顺序。没有增加算子、Python生产数值、JIT、线程、低精度或运行时数据缓存。Planner仍使用原有单调上界，因此不能声称已经改善预算准入。三份C++差异与源码只作为不可执行`.txt`证据。

## 结果与边界

CPU1，1048576行×8维、24分量、4次迭代，计入相同独立只读输入副本、普通Scheduler/NJIT拟合及六项独立输出，两代结果共存；双方定义/JIT编译排除。依次运行的三个暖态样本中位数：

|指标|实际V54|隔离C候选|暖态NJIT|
|---|---:|---:|---:|
|完整请求秒|3.227300|2.588854|10.831783|
|malloc暖峰值bytes|268464526|67181966|469774681|
|fresh RSS绝对暖峰值bytes|448626688|237731840|671285248|
|RSS相对各自输入/运行时初始基线增量bytes|201752576|524288|402817024|

候选较实际版本省约20%耗时、192MiB malloc；较NJIT约快4.18倍、少384MiB malloc。RSS与malloc分别采样；RSS基线含fixture生成后仍驻留的空闲页面，绝对RSS不能作为同等算法分配对照，更不能把524288bytes写成“整个任务只用0.5MiB”。每backend仅一个fresh malloc与一个fresh 0.5ms RSS进程，不是多轮峰值验收。大样本六项完整输出与两参考逐元素零差，精度政策不要求普遍bitwise。

第一A版两组524288行/K12中档都慢约40%；B版Eigen行tile后仍慢约33.5%（8维）和25.3%（32维）。不能把仅大样本的收益外推到全部形状；保留各版本build ID及原始样本，不将B版计时包装为C版计时。

最终C版19项直接检查通过：C/F/负stride、停止阈值相邻点、输入只读与重复结果独立、四种整输出失败前缀；实际baseline同样19项通过，四种失败跨构建逐字节一致。正常macOS权限原生9/9；关联Python529通过、3失败（总532，无skip）。失败是现有三种布局GMM检查要求普通workspace严格小于prepared，实际两侧均53440bytes；在其之前的完整值/status/旧结果与独立所有权比较已经通过。不能写成全面自审核/回归通过。初次ctest路径错误、沙箱shm拒绝导致的7/9及514/532、完整复验日志均保留，正常权限复验关闭沙箱启动问题，不关闭3项工作区失败。

## 还差什么

1. 按完整负载收紧启用条件。优先补测大NK（例如至少128MiB）在K8/12/16/24、不同D、C/F/strided及阈值前后，避免中档退化；128MiB仅下一候选阈值，尚无采用/全形状证据。
2. 复核免费arena/未来结果借用、Prepared审计、实际更短区间与子作用域；Planner必须覆盖启用阈值以下的最大旧scratch，不能只看最大N处较小的tile。未消费坏值与失败顺序契约不变；审计和物理work计入真实packing/replay。 收尾自审核另确认component密度tile到后验tile的复制尚未计入algorithm_copy_bytes，大样本为1610612736bytes（1.5GiB）。实际时间/峰值测量不受该计数遗漏影响，但不能宣称完整复制审计；优先让现有Eigen行密度直接写带行跨度的后验tile，消除整块转移，再补计尾部与packing。
3. 完整识别链、多产品/区间、thread/process/shared传输、低预算及所有权回归；合理解释3项旧工作区期望，不能直接放宽断言掩盖路线问题。
4. 候选通过后才做正常全量、ASan/UBSan/OFF、同核六轮配对及多fresh峰值，核验当前源码/wheel/worker身份，再进入实际生产路径。逻辑DAG/Excel公式未改，无本轮Excel运行验收或新增算子声明。

其他大矩阵NJIT领先差距、主要平台链、完整分配/复制审计与整体目标均保持开放。详细设计见`screening-design.md`，机器结果见`acceptance.json`及全部原始脚本、XML、失败日志。
