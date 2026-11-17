#!/usr/bin/env python3
"""Preserve exact scheduler coverage inputs and report the squeeze propagation arm."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm_directory', type=Path)
p.add_argument('executable', type=Path)
p.add_argument('profile', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
source = Path('zenjxl-decoder/src/frame/modular/transforms/mod.rs')
inputs = []
for path, name in [(a.executable, 'test-binary'), (a.profile, 'test.profraw'),
                   (source, 'transforms.rs')]:
    target = a.output / name
    shutil.copy2(path, target)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    inputs.append(dict(original=str(path.resolve()), retained=name, sha256=digest))
(a.output / 'identity.json').write_text(json.dumps(inputs, indent=2))
subprocess.run([str(a.llvm_directory / 'llvm-profdata'), 'merge', '-sparse',
                str(a.output / 'test.profraw'), '-o', str(a.output / 'test.profdata')], check=True)
with (a.output / 'coverage.json').open('x') as out, (a.output / 'diagnostics.log').open('x') as err:
    subprocess.run([str(a.llvm_directory / 'llvm-cov'), 'export', str(a.output / 'test-binary'),
                    '--instr-profile=' + str(a.output / 'test.profdata')],
                   stdout=out, stderr=err, check=True)
assert not (a.output / 'diagnostics.log').read_text(), 'coverage diagnostics require inspection'
report = json.loads((a.output / 'coverage.json').read_text())
lines = source.read_text().splitlines()
start = next(i + 1 for i, line in enumerate(lines) if '// Global inputs remain' in line)
end = next(i + 1 for i, line in enumerate(lines[start:], start) if 'buffer_info[*buf_out].grid_kind = grid_kind;' in line)
regions = {}
for unit in report['data']:
    for function in unit['functions']:
        if 'make_grids' not in function['name']:
            continue
        for region in function['regions']:
            if (region[7] != 0 or not function['filenames'][region[5]].endswith(str(source))
                    or region[0] < start or region[2] > end):
                continue
            key = tuple(region[:4])
            regions[key] = max(regions.get(key, 0), region[4])
assert regions, 'no mapped squeeze propagation regions'
summary = dict(first_line=start, last_line=end, covered=sum(v > 0 for v in regions.values()),
               total=len(regions), regions=[dict(location=k, count=v) for k, v in sorted(regions.items())])
(a.output / 'squeeze-regions.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
