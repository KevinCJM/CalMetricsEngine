# 联合矩阵请求的未知释放归因

6个独立child验证上一轮63090→f0e62。4个联合请求各5次：4×16B+1×32B，共96B；两个单协方差请求无未知释放。20次合计384B全部为epoch0已记录分配，未见过的地址为0；trace表/事件无溢出。旧新六阶段heap字段逐项相同。

实际NumPy arm64反汇编把分配返回PC df748/dd3fc对应到npy_alloc_cache_dim，释放PC c5abc对应到npy_free_cache_dim。该缓存保存维度/stride（最少2个intp，七槽），命中时不调用malloc，槽满时tailcall free。因此原epoch-only审计会把测量前缓存块的真正释放记为unknown。这些记录已找到大小及来源；不等于所有分配入口均已覆盖。

这是有额外BSS的地址诊断，不用于性能/RSS验收；96B元数据不能解释前轮数MiB的RSS差异，也不证明本轮softmax所有内存指标不增。二进制/地址及6child明细见trace-summary.json、trace-initial和numpy-*.txt。
