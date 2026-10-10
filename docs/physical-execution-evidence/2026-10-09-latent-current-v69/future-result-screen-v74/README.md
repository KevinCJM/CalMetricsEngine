# V74：未来独立结果区工作区复用初筛

本页保留初筛形成时的版本和证据。后续已经完成资格、重复对照和实际接入，见[当前实际验收](qualified-adoption/README.md)；不能把下面的初筛改称完整验收。

2026-10-09，实际a87f2c25不变，隔离候选d0f5e50f尚未采用。只改graph.cpp，借本次未写入的未来scatter结果区给full-posterior GMM，保留单次E和全部原数学内核。拒绝已写入根、prepared/observer、isolate/scopes和不够大的结果；不借输入或前次结果。

三种构建原生各9/9，正常相关Python701及新增专项16通过。首次专项4处测试假设错误及修复后16项的原始记录均保留。OFF相关6115通过，1项既有macOS舍入/库开关豁免，新增16项没有skip。普通完整回归和SAN相关Python尚在运行，不能宣称全量资格通过；通过后接续21个串行完整time/heap/RSS复测，尚无重复或RSS结论。

首组30个fresh完整time/heap进程及另2smoke通过13输出、NaN/离散/实际迭代、数学不变量、输入/旧结果hash和非别名。CPU1、六owned readonly准备、普通完整请求、两代結果保留、暖NJIT。GMM500k申请峰131,534,582→122,534,582bytes，少9,000,000bytes；耗时0.82516→0.84590秒，仍比NJIT1.88915秒省55%。约21ms/2.5%只作诊断，不用小差异单独否决候选。其余四链申请峰不变；GMM仍比NJIT多12,031,190bytes，HMM750k仍多46,540,614bytes，不能称整体内存已领先。

[设计](design.md)、[自审核](self-review.md)、[完整首组对照](comparison.json)、[原始筛选](screen.json)、[闭合复核](closeout-validation.json)。后续资格/复测/实际接入应另留完整记录，不能将本初筛改称正式验收。公开契约/Excel recipe/budget不变，没有本轮Excel重算、平台迁移、提交或推送；完整QP/前沿/随机探索及整体目标继续开放。
