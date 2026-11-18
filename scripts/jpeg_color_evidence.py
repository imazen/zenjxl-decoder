#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Retain the native JPEG color-channel regression executable, source identities and verdict."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
a.output.mkdir(exist_ok=False)
os.utime(root / 'zenjxl-decoder/src/lib.rs', None)
command = ['cargo', 'test', '--locked', '--manifest-path', str(a.manifest),
           '-p', 'zenjxl-decoder', '--features', 'jpeg', '--test', 'jpeg_scan_contract',
           '--no-run', '--message-format=json']
with (a.output / 'build.jsonl').open('x') as log:
    subprocess.run(command, stdout=log, check=True)
artifacts = [json.loads(line) for line in (a.output / 'build.jsonl').read_text().splitlines()]
executables = [row['executable'] for row in artifacts if row.get('reason') == 'compiler-artifact'
               and row.get('target', {}).get('name') == 'jpeg_scan_contract' and row.get('executable')]
assert len(executables) == 1
binary = Path(executables[0])
shutil.copy2(binary, a.output / 'test-binary')
def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()
sources = sorted({*root.glob('*/src/**/*.rs'), *root.glob('*/Cargo.toml'),
                  root / 'Cargo.toml', root / 'Cargo.lock',
                  root / 'zenjxl-decoder/tests/jpeg_scan_contract.rs',
                  *root.glob('zenjxl-decoder/tests/testdata/jpeg_scan_v012_color-*')})
rows = []
for source in sources:
    target = a.output / 'source' / source.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    rows.append(dict(path=str(source.relative_to(root)), sha256=digest(target)))
run_command = [str(binary), 'native_color_transform_controls', '--nocapture', '--test-threads=1']
identity = dict(build_command=command, run_command=run_command,
                rustc=subprocess.check_output(['rustc', '-vV'], text=True),
                binary_sha256=digest(binary), sources=rows, exit_code=None, pid=None)
record = a.output / 'identity.json'
record.write_text(json.dumps(identity, indent=2))
with (a.output / 'test.log').open('x') as log:
    child = subprocess.Popen(run_command, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, LLVM_PROFILE_FILE=str((a.output / "test.profraw").resolve())))
    identity['pid'] = child.pid
    record.write_text(json.dumps(identity, indent=2))
    result = child.wait()
assert digest(binary) == identity['binary_sha256']
for row in rows:
    assert digest(root / row['path']) == row['sha256'], row['path']
identity['exit_code'] = result
profile = a.output / 'test.profraw'
identity['profile_sha256'] = digest(profile) if profile.is_file() else None
identity['executed_tests'] = sum(int(n) for n in re.findall(r'test result: ok\. (\d+) passed;', (a.output / 'test.log').read_text()))
assert result != 0 or identity['executed_tests'] > 0, 'zero-test run is not evidence'
record.write_text(json.dumps(identity, indent=2))
print(f'Native JPEG color-channel regression: rc={result}; exact ELF {identity["binary_sha256"]}', flush=True)
sys.exit(result)
