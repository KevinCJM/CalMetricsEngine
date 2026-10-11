# 协方差首段中心化面板复用：组件候选，尚未接入

2026-10-02；本机arm64。基于代码定位真实冗余：DSYRK完整中心化后因幅度拒绝，旧路径丢弃面板，再为每个协方差单元重算中心化。临时候选仅复用首段覆盖全N、F、偶D、幅度拒绝；保留原Eigen归约树、线程恢复、对称复制和Workspace上界，其余仍原路径。参见 [原静态设计](cov-DESIGN.md)及[完整构建身份边界](cov-provenance.json)。原设计“未运行数值”是编译时状态，由本节实测更新；生产接入必须参数化唯一已有helper，不能复制临时内核。

[原生704调用](cov-native-check.log)通过、12次复用；[Python396案例](cov-correctness/cov-correctness.json)逐位通过、24次复用，覆盖C/F/补边、N余数及面板边界、常数/退化/NaN/Inf、canary、readonly和调用线程恢复。[12类、6轮组件计时](cov-timing/cov-timing.json)的新旧完整矩阵bits相同，warmup与最终结果、输入和C布局均检查；使用现有公平助手仅限制NJIT，reference TLS正常恢复，native自行准入。[线程结束记录](cov-timing/cov-thread-cleanup.json)。每次原生status检查保留，不把partition-only或编译通过称为完整请求验收。

2048×2048 F/scaled：**613.704→465.386ms**，少148.318ms/24.2%；NJIT81.847ms，仍有明显差距。C/scaled路径不变586.567→586.351ms；长序列F/correlated不适用复用188.655→189.155ms、NJIT52.959ms。0.5ms等微小正差保留但不单独否决。scaled相对NJIT原有容差外单元保留：F2644、C972；候选对原组件bits0，不能据此称与所有NJIT数值完全一致。

**尚未生产采用**：没有本候选完整Scheduler/prepared/进程验收，也没有malloc/RSS、全算子回归、新wheel或Excel重算。需要沿现有helper做唯一实现，再验证真实完整请求与内存后决定。原native候选链接的是a684 archive中与ccca逐文件相同的协方差组件，不能称完整ccca archive。初次编译漏显式arm64参数的失败保留，补参后3/3编译通过；仅本机平台。

[screening-summary.json](screening-summary.json)与manifest保留脚本/源码/日志hash；没有归档dylib/archive，不能称独立Git可重建发布证据。冻结脚本只把线程助手import改为相对邻证据目录，执行版hash单列。无需为微小差异扩展门禁框架；整体优化目标保持开放。未提交或推送。
