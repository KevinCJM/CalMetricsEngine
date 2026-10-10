"""Verify archived file integrity only; this is not runtime acceptance."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'
        and '__pycache__' not in p.parts}
assert manifest==actual,'archive file set or content changed'
print('PASS archive integrity',len(manifest),'files; no runtime/whole-goal claim')
