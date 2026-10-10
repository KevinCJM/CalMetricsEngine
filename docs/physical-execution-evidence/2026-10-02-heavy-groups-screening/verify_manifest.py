"""Verify frozen evidence digests, byte sizes and the complete file inventory."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
expected = manifest['files']
actual = {str(p.relative_to(root)) for p in root.rglob('*')
          if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
assert actual == set(expected), (actual - set(expected), set(expected) - actual)
for name, entry in expected.items():
    path = root / name
    assert path.stat().st_size == entry['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], name
print(json.dumps({'verified_files': len(expected), 'build_id': manifest['build_id']}))
