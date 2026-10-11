# V62 D: keep failure prefixes; fuse terminal separation with packet writes

Self review found that A storage loans must not include divide: scalar-first
division can fail after writing a prefix, whose remaining root bytes must
still be canonical positive zero. Keep all divide chains on original storage
and numerical routes. Add raw C++ caller-owned failure-buffer checks against
canonical divide, in C/F order and matrix/scalar operand orders. The original
row-batch route commits only complete tiles; it has a distinct failure prefix
from the unbatched writer and must be checked separately. Do not remove
initial typed-root clearing or change the canonical division implementation.

Ensure a single scalar stage also enables the existing arena-fusion context;
otherwise it writes the independent root but still allocates an unused matrix.
Signed literal expressions include a logical negate node and remain on the
existing route. Test these as genuine fallbacks; negative scalar parameters
exercise the supported leaf domain without folding away logical nodes.

When the certified terminal node already uses the Eigen scalar pair, reduce
finite/minabs over each newly written <=65536-cell piece while it is cache
resident. Report a conservative all-separated flag to this call's graph.
Only a completely separated result may skip the later certificate pass;
ambiguous pieces still cause the same canonical replay. No budget, tolerance,
operand order, output shape, division prefix or logical preparation changes.
The added optional parameters are internal C++ pair diagnostics, not an AST
operator or public Python output. Other pair users incur no certificate scan.
