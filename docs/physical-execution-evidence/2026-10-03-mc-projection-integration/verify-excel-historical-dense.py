"""Existing 185-operator Excel snapshot; explicitly excludes the ten new cases."""
import importlib.util
import json
from pathlib import Path
import sys
stage=Path(__file__).resolve().parent
root=Path('/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine')
source=root/'docs/physical-execution-evidence/2026-10-01-jacobi-direct/verify_excel_snapshot.py'
spec=importlib.util.spec_from_file_location('projection_historical_snapshot',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.INSTALL=stage/'install-dense'
module.OUT=stage/'excel-historical-dense'
prior=json.loads((module.OLD/'manifest.json').read_text())
allowed={case['case'] for case in prior['cases']}
assert len(prior['cases'])==469
sys.path.insert(0,str(module.INSTALL))
sys.path.insert(0,str(root/'tools'))
import check_excel_parity as check
original=check.write_bundle
def existing_cases(cases, destination):
    selected=((name,make) for name,make in cases if name in allowed)
    return original(selected,destination)
check.write_bundle=existing_cases
if __name__=='__main__':
    raise SystemExit(module.main())
