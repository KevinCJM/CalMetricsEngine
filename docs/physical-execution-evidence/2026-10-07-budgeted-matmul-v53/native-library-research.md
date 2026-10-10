# 本轮成熟原生库与公开方法筛选

2026-10-07。本记录为候选筛选，未将新的第三方库接入实际源码，亦未产生新的性能验收结论。现有预算修复资格与外部方法适用性分开核验。

## 已有本机证据

- 实际引擎仍是 V51；V53 位于隔离目录。V52 的大矩形完整请求相对旧 C++ 有秒级改善，但仍慢于热态同核 NJIT，且紧预算误拒绝尚待本轮修复资格闭环。
- 项目已完成 BLIS firestorm 的 AOT 单线程筛选。历史 4096 方阵的同计划 BLIS/Accelerate 为 3.553/1.087 秒，没有大额峰值节约。因此不重复将 BLIS 通用替换当作新方向；该结论仅适用于已有芯片、版本和几何。
- 本轮复核未发现标准 Homebrew 路径中的 OpenBLAS、BLIS、oneMKL 动态库。这只是路径检查，不等于系统不存在其他安装。

## 继续候选及边界

| 来源 | 可参考内容 | 本轮判断 |
| --- | --- | --- |
| [Arm 官方 macOS ArmPL 说明](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/first-macos-release-of-arm-performance-libraries) | M 系列 C/C++ BLAS/LAPACK、LP64/ILP64、顺序版及 OpenMP 版 | 顺序 LP64 版是后续库筛选候选；先固定版本、实测工作区与完整链。不能用宣传或其他芯片测量认定更快；无需先替换现有引擎。 |
| [GEMMFIP 原论文](https://arxiv.org/abs/2302.08417)、[BLIS packing 插件 API](https://github.com/flame/blis/blob/master/docs/PluginHowTo.md) | packing 与首次计算融合，避免独立复制遍历；可控制 packing 与分块树 | 可参考执行方法；当前 Accelerate 接口未暴露同等 packing 控制，不能直接移植 API 或声称实现。 |
| [BLIS 线程文档](https://github.com/flame/blis/blob/master/docs/Multithreading.md) | expert 接口的每调用 rntm_t，局部线程与循环并行控制 | 是未来可控后端的调度参考，不能改变共享 Scheduler CPU 配额。已有 BLIS 速度失败仍保留。 |
| [OpenBLAS 官方 FAQ](https://www.openmathlib.org/OpenBLAS/docs/faq/)、[运行时参数](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/) | 外层并行时控制内部线程；单线程并发构建还需 locking；区分线程和缓存参数 | 实测候选须防止嵌套并行，不能在请求中改进程全局线程设置；未完成本机替换测量。 |
| [LIBXSMM 当前官方 README](https://github.com/libxsmm/libxsmm/blob/main/README.md) | 主要小矩阵/TPP，典型几何立方根不超过约 64，默认代码生成以 JIT 为主 | 不作为当前大矩阵优先项；静态链接不证明没有请求期 JIT。若用于未来固定小块，须先证明全部内核 AOT。 |
| [M1 AMX 调度与预打包论文](https://arxiv.org/abs/2606.25426) | 多面板与常量权重预打包，其实验针对 fp32 LLM GEMM | 可参考依赖和调度方法；不把 fp32 收益外推到本项目 float64，不引入私有指令或改变误差/错误语义。 |
| [Eigen 2026-10 M4 官方比较](https://libeigen.gitlab.io/benchmarks/2026-09-apple-m4-blas-lapack-sme/) | 明确版本、同线程、结果校验、分块布局及重复测量方法 | 是测量方法参考；M4 SME 收益不能外推到当前既有 M1 测量，也不据此升级依赖。 |

## 下一轮实验顺序

1. 先完成 V53 预算策略全量、SAN、OFF 资格及原方阵控制复验；不在重构预算的同时更换数学后端。
2. 对仍有实质耗时的几何区分输入准备、验证、DGEMM、packing/归并和结果返回，避免重新优化已接近库速度的内循环。
3. 筛选 ArmPL 顺序 LP64；或者验证已有成熟库上的面板调度及不可变输入预打包。输入地址、shape、只读标志不足以证明可缓存数值，必须有内容或可信不可变版本身份；缓存常驻内存计入预算。
4. 同数据、算法、输出、核数、输入准备及结果保留方式，对现有 C++ 和预热 NJIT 测完整耗时与 fresh 峰值。保留成功与失败原始记录；只按真实的大额收益决定采用。

本轮没有新的库性能数据。普通回归/消毒器通过不等于完整性能或平台迁移通过；几 KB/几毫秒差异仍只作诊断。
