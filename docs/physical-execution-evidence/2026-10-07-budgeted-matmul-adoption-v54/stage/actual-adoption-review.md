# V54 接入前复核

范围是现有独立 MatMul 根可选物理路径；无新增算子、业务逻辑、AST/DAG/Excel数学、公开参数或结果格式。

- 当前实际目录与 V53 留存的 93 个 native source、169 个既有测试及 294 个保护文件逐个 SHA 相同；没有发现外来改动。只有 9 个 C++ 文件、2 个既有原生测试和3个新增测试/夹具需要接入。graph_native_tests.cpp 已相同，不重复改写。
- 几何资格的乘积下界用除法向上取整计算，不形成 M×K×N 溢出。M/N及leading dimension严格限于LP64的INT_MAX；K仍限8192。
- 长轴符号向量计入新库reserve，规范化预检只借用有界只读子视图；原8192 helper保护没有删除。局部范数临时容量是 min(320*K+72,2*max(M,N)+8) doubles，销毁后才增长numeric scratch。
- 数学守卫、原0.000049重放界、NaN/Inf/负零/舍入、旧原生回放和错误隔离保留。没有通过调整精度或降低库资源预留获得时间收益。
- 单份Plan只在已减至1worker仍超预算、原几何scratch更小时重选一次，清理本轮lane/worker/chunk/transport状态；逻辑图与worker cached Program不存储请求策略。标志由runtime、graph RAII和IPC24逐请求传递。
- shape/workspace审核使用同一开关；typed结果继续直接写本次独立C/F存储，结果保留、prepared和共享输入均有回归。IPC旧版本、未知mode位拒绝后健康请求恢复有原生覆盖。
- V53相同最终源码已有全量normal15876、ASan/UBSan相关2700、OFF相关2309+391条件skip及各9个native通过。实际接入后仍需重建source/wheel/worker身份，重跑native9和normal相关2700；不把隔离包通过表述为实际目录已接入验收。
- 当前90个CPU1暖态配对显示长/宽矩形相对旧实现大幅提速，但仍慢于NI。fresh24子进程峰值状态以终态为准；活跃堆基本持平，无大额内存改善。采用是修复实际既有大矩阵瓶颈，不是整体性能目标完成。

尚未提交、推送、系统安装或部署。相关性不外推到所有MatMul几何、直接算子、多根闭包、完整模型和平台迁移。
