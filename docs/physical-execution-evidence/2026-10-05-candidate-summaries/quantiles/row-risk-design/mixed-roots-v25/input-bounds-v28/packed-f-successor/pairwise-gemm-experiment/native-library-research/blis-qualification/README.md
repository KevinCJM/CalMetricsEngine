# BLIS原152项与完整筛选闭合，本机不采用该替换

固定官方2.1提交caf0db6be1202d9c83c79b51e75eceb96aa8b556，静态firestorm、单线程、64位native整数、BLAS兼容层和内部packing/small-block池关闭。源码归档SHA、许可、实际预处理宏、编译命令及二进制身份保留。原152输入hash、88准入64拒绝、Decimal80最大误差2.520e-8、C/F结果、异常拒绝、晚失败和独立两代所有权通过。

只替换分块GEMM后端，范数、精度模型、分块/归并、回放和0.000049预算保持不变。实际输出直接由BLIS typed stride API计算，无Python数值回调或全量转置。核心显式scratch上界不包含BLIS内部packing；完整fresh堆/RSS必须计入这些分配，未获得Planner总空间资格。RN标准GEMM误差模型不是供应商形式化证书。

构建两个命令均成功；初版验证器错误地扫描条件头文件文本并检查不存在的SINGLE宏，后续改用有效预处理宏核验，原失败保留；没有重新构建或修改供应商代码。后续[25组/150次时间与100fresh峰值](closed-prescreen/README.md)已经闭合；本机全部五组慢于Accelerate同计划，未带来大额内存收益，因此不采用该替换。未接入引擎，整体目标未完成。
