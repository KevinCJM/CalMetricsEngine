# 单根高分位数短尾选择

2026-10-02，基线 ccca6b67a5220ea2134a4743d281b80c5af4c096e7d41c3d5b747bb8de70e331。仅优化 ordered_stat 的现有数值选择路径；共用 reduce_value 和 reduction 调用。共享多根缓存仍用完整排序，不写缓存，也不新增公开算子。

现有 count>=1024、无 NaN、无混合零符号守卫不变。需要相邻排名且 upper>count/2 时，以原 std::nth_element 选 lower，再以 std::min_element 扫短尾补 upper；其余保留原 upper/max-prefix。每个排名与完全排序相同，相等比较值在守卫内 bits 相同，原插值、溢出、Infinity*0、错误与计数不变。偶数 median 保持原分支。没有新浮点运算、并发、ISA、库、缓存或所有权变化。

每次读取当前消费数据；原只读输入复制到已有独占 scratch，mask count 校验、源 N Planner 上界和 Workspace 高水位容量不变。分配/复制/IPC19 不变，不变算法工作区仍计入 memory。Excel 公式、数学版本、预算与导出依赖闭包不变；以受影响原生回归及历史快照复比确认等价，本轮无新 Microsoft Excel 重算。

先以真实候选 wheel 做完整请求串行均衡计时，与实际 ccca wheel 及 warmed NJIT 同一 CPU 预算、独立结果寿命比较；覆盖大随机、高 q、有序/逆序/等值、低 q/中位数/短输入与 masks、prepared。候选未验收前不宣称采用；微小正差仅诊断，中大型实质差距继续开放。若收益不足，只撤销本次 reduction/test 精确补丁，保留共享脏区其他改动。
