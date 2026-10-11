# 两项生产优化：最终只读代码审核

结论：当前文件未发现尚未解决的阻断问题。首次发现的测试收集阻断已由 root 修复。
本审核只覆盖下列 before→production 差异与两份新测试；没有修改仓库文件、编译或执行数值/测试/基准。
CodeGraph 先导航，源码及 before 文件随后逐项对照；AGENTS/D1-D8/Excel X1、X10 的既有纪律沿用。

## 已修复的可执行发现

- **[P1，已修复] 测试绝对导入阻断标准收集。** 新 tests/test_order_linear_packing.py:8 初稿用
  `from test_order_selection_library import ...`。pyproject.toml:84 默认 importlib 收集，
  该顶层名称没有由项目配置注入；应使用相对 `.test_order_selection_library`。
  root 反馈首次 collection 失败日志保留，并已修复；本 agent 重新读取当前第8行确认相对导入。
  这项是测试可执行性问题，不是生产数值缺陷。修复后的收集/运行结果由 root 统一验证。

## Cholesky 分组更新

对照：stage/before-cholesky_library.hpp → cpp/operators/cholesky_library.hpp。

- :43 完整tile的 j 从0升序递增8，并且始终 j<=common_count；只有 common_count-j>=8 才读取8值。
- :89–95 四行准入 count-r>=4，故 r+3<count。共同长度 L=begin+r-k-8+1；begin为16倍数且k<=begin-8，L>0。
  第s行原长度恰是L+s；:60 尾部 L+s-j 精确补全共同不足8元素和各行独有0..3值。
  最后列为该行 begin+r+s<=n-1；不读 padding，不丢尾部，不发生 unsigned 减法下溢。
- :75–88 原因素阶段逐字保留，先完成全部因素/镜像，再处理独立目标。
  新 :47–52 每目标仍u=0..7逐项multiply/subtract；shared源为已生成的upper mirror，目标在独占panel，更新不改源。
  原true division、finite、sqrt、copy_n/finish、R16/K8和n+17 pitch未变，没有零项跳过、横向归约或FMA语义。
- scalar/generic路径未扩大准入：matrix_foundation.cpp:283–290保留macOS15+arm64/Eigen/n>=1024及overflow检查；
  :454仅auto ISA进入panel，:469原canonical处理保留。C/F/负/补边输入仍由原a.at借用读取。
- :460–465 原numerical replay保留，只重放DOMAIN_ERROR/NON_FINITE_RESULT；资源错误不吞掉。
  原地 covariance pair 的producer恢复责任与F成功发布三角、observer/scope降级未被该diff触及。
- 数值heap工作区仍16*(n+17)+128 doubles，4096阶527,488 B；两份下三角复制审计未变。
  固定 values[4]+shared逻辑容量320 B，不是动态数组；实际寄存器/栈帧以及库编译行为仍需root构建证据。

## 顺序统计线性打包

对照：stage/before-reduction.cpp → cpp/operators/reduction.cpp。

- :56–57 只让rank>0且逻辑C连续的输入/可选mask使用原始指针，保留旧scratch的逻辑C消费顺序。
  c_contiguous()是operators.hpp:138–157的轴级判据，忽略singleton无影响stride；不得改成dense_contiguous。
  F、负、间隔或任一不连续mask在:68–71保留x.f/mask.u。内部inline scalar仍走Value.f，未解引用其null data。
  公开median/quantile仍由registry.cpp:195要求rank1..3，未新增标量输入能力。
- registry.cpp:522在任何workspace/数值打包前调用validate_data_structure；:424–433完整检查全部逻辑mask字节<=1。
  两个打包分支都先判断选择再append值；未选中NaN/Inf/负零不参与has_nan/zero守卫，也不扩大消费域。
- :75–76以实际append count记录count*8字节，包括empty mask的0；删除旧重复mask计数扫描。
  :77不足样本失败仍在out.set之前。:259–263直接调用等价的ordered_stat只减少委托/重复扫描；
  q约束、NaN/混合零sort守卫、nth/max、插值顺序和Infinity/overflow行为与before完全相同。
- Workspace仍按输入x.size预算，读入scratch后选择，不改借用输入。Audit*只在匿名私有函数新增，公开reduce_value/registry签名不变。
- graph.cpp:384–427多根full-sort cache未改，仍按本次generation重建；没有把partition冒充有序缓存。
  本轮不新增内容缓存，prepared同地址数据/mask每次重读。不同请求和保留快照不成为数值工作区。

## 新测试与验证边界

- tests/test_cholesky_grouped_library.py：指定1026/1027/1028/1032/1039/1040用稠密SPD覆盖2/3/4/8/15/16尾行、C/F/双负/NaN补边。
  cancellation/subnormal/负零明确位于(20,8)共同tile；四组共享tile溢出检查canonical早坏主元优先级；
  NaN/Inf验证先输入校验；C输出canary/input位模式及实际F prepared失败后恢复/旧快照独立均有断言。
  未把直接out异常误写为全矩阵事务性不变；仅检查边界不越界与输入/旧独立结果不变。
- tests/test_order_linear_packing.py：C/F/不同mask布局、负stride、rank2/3 singleton特殊stride、选中数1023/1024/1025、
  被屏蔽sNaN/负零、实际NaN payload/零sign、有界native共享排序oracle、empty及非法mask、公开rank0拒绝均覆盖。
  旧快照在当前mask数量改变、失败、下一borrowed run后仍独立；多根full-sort和masked独立copy计数并存。
  NaN oracle只取整数rank，避免先前fractional不同payload歧义；有限插值另按原表达式核验。

上述属于源码/测试设计审核，不是运行通过声明。不以root已有native9/9或候选ABI结果替代新wheel完整测试。
实际FMA缺失、栈帧、malloc/RSS、普通/prepared性能与成熟库路径身份由root串行验收；本记录不承诺达到NJIT目标。
当前差异没有新增公开算子、dtype/axis/layout/数学或Excel公式契约，也没有新增私有线程、数值缓存或全矩阵缓冲。

## 审核时身份

- `cpp/operators/cholesky_library.hpp`: `d7c91b3be3e49ee3b30ab619d869b04ebe36e3a41e4204a11e4ae229bfa075a9`
- `cpp/operators/reduction.cpp`: `0024db411c68609c6f64c6c4478da9496c5b99a93c5846b3d5ad70c38f1aedf0`
- `tests/test_order_linear_packing.py`: `330a681bbf081bb7867003f4d83d85ecf5fe4ce1663470ef66a6903f45ad1186`
- `tests/test_cholesky_grouped_library.py`: `dc85730cc376397390d217bc4a75b202f477093849552a3eba0df3d60b206b77`
- before-cholesky_library.hpp: `7916356c8810898c5fa7ada5df8b2ffa2822be3188233924c8b5590a9a530d1a`
- before-reduction.cpp: `d4096a7dc718dea263c02ee53bbcbf9ed3de94f2276b1c598a98f827d9aa7859`
