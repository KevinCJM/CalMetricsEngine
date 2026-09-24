"""Controlled short-batch lane costs; forced lanes are diagnostics, not defaults."""
import argparse
from contextlib import ExitStack
import json
import statistics
import time
import numpy as np
from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig, SharedInputBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    graph = GraphCompiler({'x': {'kind': 'tensor', 'axes': ['scenario','path','asset'],
                                'shape': ['S','P','N']}}).compile({'value':'x*2+1', 'mask':'x*2+1>3'})
    x = np.full((252,16,8), 8.)
    x.flags.writeable = False
    starts, ends = np.zeros(8, np.int64), np.ones(8, np.int64)
    report = {'schema':'transport-diagnostic-1', 'cpu_budget':4, 'rows':8,
              'input_shape':list(x.shape), 'results':[]}
    for lane in ['single','thread','process_inline','process_shared','process_preshared']:
        config = PlannerConfig(thread_work_units=1 if lane == 'thread' else 1e100,
            process_work_units=1 if lane.startswith('process') else 1e100,
            shared_memory_threshold_bytes=1 if lane == 'process_shared' else 1 << 30,
            min_rows_per_worker=1, max_processes=4)
        with ExitStack() as stack:
            inputs = {'x':x}
            if lane == 'process_preshared':
                inputs = stack.enter_context(SharedInputBundle.from_inputs(inputs))
            engine = stack.enter_context(AdaptiveScheduler(cpu_budget=4, config=config))
            def run():
                return engine.execute(graph, inputs, starts, ends)
            for _ in range(5):
                result = run()
            for actual in result.outputs[0].values:
                np.testing.assert_array_equal(actual, x*2+1)
            for actual in result.outputs[1].values:
                np.testing.assert_array_equal(actual, np.ones(x.shape,bool))
            samples = []
            for _ in range(21):
                begin = time.perf_counter_ns()
                for _ in range(4):
                    result = run()
                samples.append((time.perf_counter_ns()-begin)/4)
            row = {'requested_lane':lane, 'median_ns':statistics.median(samples),
                   'samples_ns':samples, 'audit':result.audit}
            report['results'].append(row)
            print(lane, round(row['median_ns']/1e6,3), 'ms', flush=True)
    with open(args.output,'w') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')

if __name__ == '__main__':
    main()
