# vvexpm1：当前不采用的静态边界

2026-10-03。只读，不改源码、不运行数值/编译/测试/性能。root已明确本阶段不采用未经证明std::expm1逐位等价的vForce；下一性能候选仅评估Eigen affine的独立asset lanes。

已核对本机Apple SDK `.../Accelerate.framework/Frameworks/vecLib.framework/Headers/vForce.h:240–256` 有double接口 `vvexpm1(double*,const double*,const int*)`（macOS10.7+）。其函数说明强调近零精度，但通用说明`:27–45`明确结果/denormal处理可能因OS和microarchitecture不同、异常flag不保证。没有在已读声明中找到in-place overlap保证。因此API存在不构成std::expm1逐位、全部round模式或重叠安全证据。

当前 `cpp/operators/transcendental_accelerate.cpp` 仅为exp/log实现受控Apple/Isa/FE_TONEAREST/输入域与stride/tile候选，不能直接把expm1加入后复用同一数值域。expm1 near-zero、负零、round-to−1边界与大正overflow有自己的契约。

若以后独立screen，合法物理形态是：保持当前同一个ascending-factor affine fold，按bounded tile的独立结果调用vvexpm1；不改initial、factor归约、weight-first乘、贡献跨tile顺序。count要经INT_MAX和tile容量检查。若未证明重叠，使用独立有界input/output tile并计高水位/复制，不能默认in-place。输入只读、公开root独立所有权、privatepool禁止和现有Scheduler预算不变。

关键采用障碍不只是最终allclose：

- period<=−1、NAV正值/overflow、tail<=q05、target>=、sort/quantile及首故障分支都是离散契约；几个ULP可能改变这些结果。
- FE_TONEAREST和finite/normal guard不足以证明上述阈值远离边界；当真实caller可编辑weights/target时，不能凭固定fixture边界安全推广任意DAG。
- 每元素同时执行std::expm1来校验再选择虽可作为诊断，但不能宣称省去原libm计算或完整加速。
- 极端值的scalar replay必须在tile任何消费者更新前完成，避免重放后双计contribution；不同OS/构建identity要重新证据化。

未来如有授权的新数值容差/实现版本，再冻结output IEEE/ULP域、threshold fallback及root离散exact范围，并执行实际完整单/多组合对照和四种round模式、NaN/Inf、subnormal/±0、nextafter(−1相关输入)、overflow边界、tile/canary/ownership测试。库内部allocation与workspace仍需heap/RSS补充，链接Accelerate不能当runtime hit。

当前结论：保留成熟标准库std::expm1为唯一当前数值职责；vvexpm1作为未采用、未数值验证候选记录。没有性能结论，也不为此创建新框架。
