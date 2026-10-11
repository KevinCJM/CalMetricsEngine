"""Editable existing-operator graphs for three finite-input portfolio risks.

This is a mathematical fixture, not platform sampling/repair/service migration.
Platform invalid-input policy remains a caller concern and must be qualified
against the unchanged canonical graph before any physical lowering is admitted.
"""
TYPES=dict(w=dict(kind='matrix',axes=['asset','factor'],shape=['N','K']),
           r=dict(kind='matrix',axes=['time','factor'],shape=['T','K']),
           p=dict(kind='vector',axes=['component'],shape=['1']))
Y='matmul(w,transpose(r))'
QT=f'quantiles_axis({Y},p,1)'
# Reduce a declared singleton probability dimension before broadcasting on N.
THRESHOLD=f'transpose(broadcast_like(sum_axis({QT},1),transpose({Y})))'
TAIL=f'less_equal({Y},{THRESHOLD})'
TAIL_COUNT=f'sum_axis(where({TAIL},1,0),1)'
CVAR=f'-divide(sum_axis(where({TAIL},{Y},0),1),maximum({TAIL_COUNT},1))'
NEGATIVE=f'less_than({Y},0)'
NEGATIVE_COUNT=f'sum_axis(where({NEGATIVE},1,0),1)'
NEGATIVE_MEAN=f'divide(sum_axis(where({NEGATIVE},{Y},0),1),maximum({NEGATIVE_COUNT},1))'
CENTER=f'transpose(broadcast_like({NEGATIVE_MEAN},transpose({Y})))'
DELTA=f'subtract({Y},{CENTER})'
SQUARED=f'sum_axis(where({NEGATIVE},multiply({DELTA},{DELTA}),0),1)'
DOWNSIDE=f'where(greater_than({NEGATIVE_COUNT},1),sqrt(divide({SQUARED},maximum(subtract({NEGATIVE_COUNT},1),1))),0)'
NAV=f'cumulative_product_axis(add(1,{Y}),1,1)'
PEAK=f'cumulative_max_axis({NAV},1,1)'
DRAWDOWN=f'max_axis(subtract(1,divide({NAV},{PEAK})),1)'
ROOTS=[CVAR,DOWNSIDE,DRAWDOWN]

def compile_graph(compiler,*,proof_counts=False,output_order='C'):
    roots=ROOTS+[TAIL_COUNT,NEGATIVE_COUNT] if proof_counts else ROOTS
    return compiler(TYPES).compile(roots,result_format='typed',output_order=output_order)
