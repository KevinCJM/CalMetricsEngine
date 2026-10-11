# 候选096e自审与边界

同源正常wheel/native/worker88项身份已匹配；726项相关通过（新增16项1537/2049阶、RHS1/64、C/F、private/input、两代所有权和真实workspace断言）。4个完整时间及16个heap/RSS任务串行终态0，所有独立原方程、输入只读及两代结果检查通过。全部原始轮次保留；浮点最大差约4.34e-19，canonical首错/1e-14/singular契约不放宽。

固定512列按Netlib顺序调度成熟DGETRF2/DLASWP/DTRSM/DGEMM。基于已证N1024..4096/K1..64，所有int和offset均有界；pivot调整和left/right swaps保留。先INFO、factor有限与canonical证书，再DGETRS。factor在DGETRS后无后续消费者，才改作residual panel和3NK数组，容纳上界(64+3K)N<=256N<N²；RHS及5N证书使用不重叠独占Workspace。下游原系数从retained原只读A或已证persistent leaf producer重建；失败重建/回放保持同一canonical数学及实际audit。输出仅在残差通过后写独立C/F根，当前输入、公共矩阵和过去根不写入。

Planner query为NK+5N；copy query额外含N²和512N vendor/ABI估计，显式F canonical回放的N²仍保守包含。没有新算子、数值缓存、Python内核、隐式线程或Excel公式/预算/IPC变化。新栈/代码内存及allocator驻留不能用requested heap替代。

完整任务结果：直接C/F约0.573/0.476秒（NI0.692/0.579）；公共F0.485/0.681；privateF0.490/0.651。privateF warm申请峰676.055/809.539MB，少约133MB。retained申请峰仍多0.47–0.73MB，RSS仍多约11–21MB，不能宣称全部内存通过。相对ac54 requested峰减少约16.5MB，但因子时间增加约30–60ms，不能称时间和内存同时全面优化。root924未采用本候选；normal全量12,674、真实ASan/UBSan与OFF相关各726/原生各9的原始终态及88项同源身份已验证，全部零失败/错误/skip。独立operator、非DD、更大矩阵及其他主要计算链仍开放。
