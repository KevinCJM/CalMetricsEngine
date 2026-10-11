"""Validate recorded identities, ownership, counters and performance samples."""
import hashlib
import json
import statistics
from pathlib import Path

S = Path(__file__).resolve().parent
load = lambda p: json.loads((S / p).read_text())
builds = {'old': load('baseline.json')['build_id'], 'new': load('final-identity.json')['build_id']}
timing = load('timing/timing.json')
assert len(timing) == 11
by_label = {r['case']['label']: r for r in timing}
for row in timing:
    n = row['case']['n']
    assert row['correctness'] and row['strict_njit_equivalent']
    assert len(row['rounds']) == 9 and row['cpu_budget'] == 1
    for backend in ('old', 'new', 'njit'):
        identity, audit = row['identities'][backend], row['audits'][backend]
        assert identity['input'] == row['identities']['old']['input']
        assert identity['independent_results'] and audit['input_unchanged']
        assert statistics.median(row['samples'][backend]['wall_ns']) / 1e6 == row['median_ms'][backend]
        assert len(row['samples'][backend]['wall_ns']) == (45 if n < 256 else 9)
        verified = load('timing/validated-' + row['case']['label'] + '-' + backend + '.json')
        assert verified['input'] == identity['input']
        assert verified['output_sha256'] == audit['output_sha256']
        if backend != 'njit':
            assert audit['engine_build_id'] == builds[backend]
            assert audit['cpu_budget'] == audit['cpu_tokens'] == 1
            assert audit['lane'] == 'single'
            assert audit['python_fallback'] == audit['python_operator_calls'] == audit['request_time_compilation'] == 0
            count = n * (4 if row['case']['name'] == 'solve_multi' else 1)
            assert sum(c['algorithm_copy_bytes'] for c in audit['native_chunks']) == (n*n + count)*8
    assert row['identities']['new']['residuals']['changed_bits'] == 0
    assert row['audits']['old']['output_sha256'] == row['audits']['new']['output_sha256']
assert all(r['stopped'] for r in load('timing/timing-worker-cleanup.json').values())
memory = load('memory/memory.json')
assert len(memory) == 12
untracked = []
for row in memory:
    label, backend = row['case']['label'], row['backend']
    timing_row = by_label[label]
    assert row['input'] == timing_row['identities'][backend]['input']
    assert row['audit']['input_unchanged']
    assert row['audit']['output_sha256'] == timing_row['audits'][backend]['output_sha256']
    if backend != 'njit':
        assert row['audit']['engine_build_id'] == builds[backend]
    if row['kind'] == 'heap':
        assert row['heap_schema'] == 'continuous-live-1'
        assert row['cold_peak_bytes'] == row['cold_heap']['peak_bytes']
        assert row['warm_peak_bytes'] == row['warm_heap']['peak_bytes']
        for stage, stats in row.items():
            if isinstance(stats, dict) and 'table_overflows' in stats:
                assert stats['table_overflows'] == 0
                if stats['untracked_frees']:
                    # Preserve the observed post-peak release limitation, do not erase it.
                    assert stage == 'warm_retained_heap' and backend != 'njit' and stats['untracked_frees'] == 2
                    untracked.append({'case': label, 'backend': backend, 'stage': stage, 'count': 2})
    else:
        assert row['rss_schema'] == 'setup-through-warm-monitor-control-1'
        assert row['rss_idle_delta'] == 0
        assert row['rss_delta'] == row['rss_peak'] - row['rss_baseline']
assert len(untracked) == 4
for label in ('medium2048_c', 'large4096_f'):
    heaps = {r['backend']: r for r in memory if r['case']['label'] == label and r['kind'] == 'heap'}
    for key in ('cold_peak_bytes', 'warm_peak_bytes'):
        assert heaps['old'][key] == heaps['new'][key]
identity = load('final-identity.json')
root = Path('/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine')
assert len(identity['source']) == 77
for path, expected in identity['source'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
report = dict(status='PASS_WITH_DISCLOSED_TRACKING_LIMITATION', timing_cases=11, rounds=9, fresh_memory_children=12,
              heap_stages=36, source_inputs=77, old_new_bitwise_equal=True, strict_njit_equivalent=True,
              cold_warm_tracking_overflow_or_untracked_frees=0, post_peak_untracked_frees=untracked,
              limitation='Two untracked frees per native child occur only when deleting the final result after warm peak capture; origin not resolved. No complete lifetime-allocation claim.')
(S/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
