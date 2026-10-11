# V3 分离通用轴路径

V2完整请求测出32768×64 F布局回退（普通25.93→49.13ms，direct20.00→30.02ms），未删除该结果，也未简单归为噪声。V3把库行driver从通用模板完全拆开，恢复原axis_reduce源码；恢复前通过整体文件SHA核对原baseline，再仅增加独立driver与准入。

正式对象与同flags独立对象逐字节一致。通用axis恢复224B；库driver272B+行helper192B=464B。latent_execute仍1088B且分别于0x6cec/0x6d7c完全恢复后tailcall库/通用路径，不叠加。库局部链比原224B增加240B，不能藏在workspace=0里；已知共同入口峰不增，系统数学函数/异常/外层栈未穷尽，完整内存仍须实际heap/RSS。

数学准入、错误写入顺序、连续axis几何和独立output均沿V2契约；无新scratch、pack、线程、协议、算子或Excel公式。V2静态同行审核可复用不变helper与guard，V3分派/loop由主代理复核，并用新构建重新跑回归和公开性能。
