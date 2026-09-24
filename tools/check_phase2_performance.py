"""Run the Phase-2 paired C++ vs NJIT performance acceptance gate.

This is intentionally not part of ordinary unit-test CI: it requires a local
BetterSaaTaa checkout plus Numba and is sensitive to host load.  It executes the
actual benchmark against the current source tree, then fails unless the paired
Native/NJIT ratio stays below the configured ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--better-root",
        type=Path,
        default=Path("/Users/chenjunming/Desktop/KevinGit/BetterSaaTaa"),
    )
    parser.add_argument("--products", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--history", type=int, default=2520)
    parser.add_argument("--intervals", type=int, default=12)
    parser.add_argument("--micro-histories", nargs="+", type=int, default=[63, 252])
    parser.add_argument("--cpu-budget", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.90,
        help="Maximum accepted prepared Native/NJIT paired median ratio.",
    )
    parser.add_argument(
        "--max-scheduler-ratio",
        type=float,
        default=1.00,
        help="Maximum accepted ordinary Scheduler.execute/NJIT paired median ratio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".build-phase2/performance-gate"),
    )
    args = parser.parse_args()

    if not 0 < args.max_ratio < 1:
        parser.error("--max-ratio must be between 0 and 1")
    if not 0 < args.max_scheduler_ratio <= 1:
        parser.error("--max-scheduler-ratio must be between 0 and 1")
    if not args.better_root.is_dir():
        parser.error(f"BetterSaaTaa root does not exist: {args.better_root}")
    if (
        min(
            *args.products,
            *args.micro_histories,
            args.history,
            args.intervals,
            args.cpu_budget,
            args.repeats,
        )
        < 1
    ):
        parser.error("all workload dimensions must be positive")

    script = Path(__file__).with_name("benchmark_phase2_graph.py")
    micro_script = Path(__file__).with_name("benchmark_phase2_micro.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    records = []
    for products in args.products:
        output = args.output_dir / f"{products}x{args.intervals}x16.json"
        command = [
            sys.executable,
            str(script),
            "--better-root",
            str(args.better_root),
            "--products",
            str(products),
            "--history",
            str(args.history),
            "--intervals",
            str(args.intervals),
            "--cpu-budget",
            str(args.cpu_budget),
            "--repeats",
            str(args.repeats),
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        payload = json.loads(output.read_text())
        paired = payload["paired_native_auto_vs_njit"]
        ratio = float(paired["right_over_left"])
        record = {
            "products": products,
            "njit_ms": float(paired["left_median_ms"]),
            "native_ms": float(paired["right_median_ms"]),
            "ratio": ratio,
            "max_ratio": args.max_ratio,
            "passed": ratio <= args.max_ratio,
        }
        records.append(record)
        print(
            f"{products} products: Native {record['native_ms']:.3f} ms / "
            f"NJIT {record['njit_ms']:.3f} ms = {ratio:.3f} "
            f"(required <= {args.max_ratio:.3f})"
        )
        if not record["passed"]:
            failed.append(record)

    micro_output = args.output_dir / "micro-1x1x5.json"
    subprocess.run(
        [
            sys.executable,
            str(micro_script),
            "--better-root",
            str(args.better_root),
            "--histories",
            *(str(value) for value in args.micro_histories),
            "--cpu-budget",
            str(args.cpu_budget),
            "--rounds",
            str(max(101, args.repeats * 40 + 1)),
            "--output",
            str(micro_output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    micro_payload = json.loads(micro_output.read_text())
    for item in micro_payload["records"]:
        ratio = float(item["native_over_njit"])
        scheduler_ratio = float(item["scheduler_over_njit"])
        record = {
            "products": 1,
            "intervals": 1,
            "metrics": 5,
            "history": int(item["history"]),
            "njit_us": float(item["njit_median_us"]),
            "native_us": float(item["native_median_us"]),
            "ratio": ratio,
            "max_ratio": args.max_ratio,
            "scheduler_us": float(item["scheduler_median_us"]),
            "scheduler_ratio": scheduler_ratio,
            "max_scheduler_ratio": args.max_scheduler_ratio,
            "passed": (ratio <= args.max_ratio and scheduler_ratio <= args.max_scheduler_ratio),
        }
        records.append(record)
        print(
            f"1 product / 1 interval / 5 metrics / {record['history']} obs: "
            f"Prepared {record['native_us']:.3f} us / NJIT {record['njit_us']:.3f} us "
            f"= {ratio:.3f} (required <= {args.max_ratio:.3f}); "
            f"Scheduler {record['scheduler_us']:.3f} us / NJIT "
            f"= {scheduler_ratio:.3f} (required <= {args.max_scheduler_ratio:.3f})"
        )
        if not record["passed"]:
            failed.append(record)

    summary = args.output_dir / "gate-summary.json"
    summary.write_text(json.dumps(records, indent=2) + "\n")
    if failed:
        details = ", ".join(
            (
                f"{item['products']} products={item['ratio']:.3f}"
                if item.get("products") != 1 or item.get("metrics") != 5
                else f"1x1x5/{item['history']} prepared={item['ratio']:.3f}, ordinary={item['scheduler_ratio']:.3f}"
            )
            for item in failed
        )
        raise SystemExit(
            f"Phase-2 performance gate failed: {details}; prepared required <= {args.max_ratio:.3f}, "
            f"ordinary required <= {args.max_scheduler_ratio:.3f}"
        )

    print(f"PASS: all {len(records)} workloads beat NJIT by the configured margin.")


if __name__ == "__main__":
    main()
