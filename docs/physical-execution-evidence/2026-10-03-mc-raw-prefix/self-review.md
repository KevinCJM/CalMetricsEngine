# 自审核与收尾核对

按当前 AGENTS、数学/Excel 纪律和源码复核；静态审核与数值执行分开。
最终生产身份85d2003f，比较基线97fdfe52，未提交或推送。

- 原始带匹配是私有依赖闭包，没有公共算子/Excel公式或金融语义变化。
  scalar/数组、χ/df、不同I/F、公开中间根、selected roots及isolate退回契约保持。
- 旋转(+0)、旋转乘scale、均值左加、资产初态及升序factor fold保持原树。
  保留canonical scale pass的错误顺序，不以数值回退掩盖失败。
- fullbatch比较含资产贡献状态，避免乘积溢出；actual workspace仍checked add/mul。
  同一binder参与预算与执行，interval结构/结果slot先预检；数值仍按interval执行。
- rank优先仅修复空数组描述；容量仍逐节点取最大size，没有放宽runtime广播。
  通用sum空样本失败、count_true空mask零、sum_axis空轴零分别按各自契约验收。
- raw输入/结果所有者保持；实际pack为copy子计数。worker IPC22与Program v7/v8区分。
  CPU实际为1，没有竞争库池、Python回调/数值回退或请求JIT。
- 六对时间、六对heap及一对RSS的范围/输入/参考签名经独立静态复核。
  旧/新时间配对参考相同；实际峰值与累计分配分开，所有allocator/RSS盲点保留。
- 12,267个当前node IDs由无重复PASS集合精确覆盖；120空轴最终文本SHA6c94。
  源87、installed2、wheel和当前244输入快照SHA吻合；97/105历史身份分别核对。
- 独立静态审核未发现新增生产缺陷；已修复§9及roadmap中的陈旧当前/历史表述。
  11份相关Markdown、339个本地链接/锚点和围栏检查通过，两份新Python测试Ruff通过，
  git diff --check通过。CodeGraph两次同步均成功，最后一次含最终空白修正。

公开ownership用例补正式driver的旧typed scalar值核查；NaN只核分类，
prepared旧metadata逐代全量保存未覆盖，不夸大生命周期证据。
固定只读replay参考每调用复制raw，原owned RNG不付这项额外copy，
仍须以真实owned入口复证，不能用本次结果关闭整个平台目标。

原始沙箱、空rank、成本夹具和empty oracle失败均保留。
初版错误empty夹具只有SHA和日志，未保存完整体；修正7add及最终6c94体
按[版本清单](diagnostic/empty-fixture-versions.json)区分，避免混用历史与当前测试身份。
manifest校验仅证明归档字节完整；完整随机/业务/服务及新Microsoft Excel验收仍开放。
