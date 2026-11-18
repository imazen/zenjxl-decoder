#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Retain exact JPEG scan coverage objects and report only the changed source regions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--report-only', action='store_true')
args = parser.parse_args()
if not args.report_only:
    args.output.mkdir(exist_ok=False)

def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

if not args.report_only:
    command = ['cargo', 'test', '--locked', '-p', 'zenjxl-decoder',
               '--features', 'jpeg', '--lib', '--test', 'jpeg_scan_contract',
               '--no-run', '--message-format=json']
    with (args.output / 'build.jsonl').open('x') as log:
        subprocess.run(command, stdout=log, check=True)
    rows = [json.loads(line) for line in (args.output / 'build.jsonl').read_text().splitlines()]
    binaries = {row['target']['name']: Path(row['executable']) for row in rows
                if row.get('reason') == 'compiler-artifact' and row.get('executable')
                and row['target']['name'] in ['zenjxl_decoder', 'jpeg_scan_contract']}
    assert set(binaries) == {'zenjxl_decoder', 'jpeg_scan_contract'}
    identity = {'build_command': command, 'rustc': subprocess.check_output(['rustc', '-vV'], text=True),
                'objects': [], 'runs': [], 'sources': []}
    for source in sorted({*Path('.').glob('*/src/**/*.rs'), *Path('.').glob('*/Cargo.toml'),
                          Path('Cargo.toml'), Path('Cargo.lock'),
                          Path('zenjxl-decoder/tests/jpeg_scan_contract.rs'),
                          *Path('.').glob('zenjxl-decoder/tests/testdata/jpeg_scan_v012*')}):
        target = args.output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        identity['sources'].append({'path': str(source), 'sha256': digest(source)})
    for name, binary in binaries.items():
        shutil.copy2(binary, args.output / name)
        identity['objects'].append({'path': str(binary), 'retained': name, 'sha256': digest(binary)})
    for index, (name, filters) in enumerate([
            ('zenjxl_decoder', ['baseline_trailing_zrl_']),
            ('jpeg_scan_contract', []),
        ]):
        env = os.environ.copy()
        env['LLVM_PROFILE_FILE'] = str(args.output.resolve() / f'run{index}-%p-%m.profraw')
        run = [str(binaries[name]), *filters, '--nocapture', '--test-threads=1']
        with (args.output / f'run{index}.log').open('x') as log:
            result = subprocess.run(run, stdout=log, stderr=subprocess.STDOUT, env=env)
        text = (args.output / f'run{index}.log').read_text()
        count = sum(int(n) for n in re.findall(r'test result: ok\. (\d+) passed;', text))
        assert result.returncode == 0 and count > 0, (run, text)
        identity['runs'].append({'command': run, 'exit_code': result.returncode, 'executed_tests': count})
    for row in identity['sources']:
        assert digest(Path(row['path'])) == row['sha256'], row['path']
    for row in identity['objects']:
        assert digest(Path(row['path'])) == row['sha256'], row['path']
    profiles = sorted(args.output.glob('*.profraw'))
    assert len(profiles) == len(identity["runs"])
    identity['profiles'] = [{'path': p.name, 'sha256': digest(p)} for p in profiles]
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
else:
    identity = json.loads((args.output / 'identity.json').read_text())
    for row in identity['sources']:
        assert digest(Path(row['path'])) == row['sha256'], row['path']
    for row in identity['objects']:
        assert digest(args.output / row['retained']) == row['sha256'], row['path']
    for row in identity['profiles']:
        assert digest(args.output / row['path']) == row['sha256'], row['path']
# Derived reports always get a fresh directory; preserve earlier exporter failures.
for report_index in range(10000):
    report_output = args.output / f'report-{report_index}'
    try:
        report_output.mkdir()
        break
    except FileExistsError:
        continue
else:
    raise RuntimeError('no fresh report directory available')
print('Derived coverage report: ' + str(report_output), flush=True)
sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], text=True).strip())
llvm = sysroot / 'lib/rustlib/x86_64-unknown-linux-gnu/bin'
# Unit-test and ordinary library instances can have different mappings with
# identical symbol names. Bind profiles to each exact object before exporting.
report = {'data': []}
for name, indices in [('zenjxl_decoder', [0]), ('jpeg_scan_contract', [1])]:
    profiles = [path for index in indices for path in args.output.glob(f'run{index}-*.profraw')]
    profile = report_output / (name + '.profdata')
    subprocess.run([str(llvm / 'llvm-profdata'), 'merge', '-sparse', *map(str, profiles),
                    '-o', str(profile)], check=True)
    target = report_output / (name + '.mapped.coverage.json')
    diagnostics = report_output / (name + '.mapped.diagnostics.log')
    dump = report_output / (name + '.mapping-dump.txt')
    with dump.open('x') as out, diagnostics.open('x') as err:
        subprocess.run([str(llvm / 'llvm-cov'), 'export', '--num-threads=1', '--dump', str(args.output / name),
                        '--instr-profile=' + str(profile)], stdout=out, stderr=err, check=True)
    diagnostic_text = diagnostics.read_text()
    # LLVM 23 reports missing zero-hash records for optimized dependency
    # functions. Keep their exact names; none may belong to the decoder.
    missing = re.findall(r"hash-mismatch: No profile record found for '([^']+)' with hash = (0x[0-9a-f]+)", diagnostic_text)
    permitted = ['_8bytemuck', '_9byteorder', '_10num_traits', '_19safe_unaligned_simd', '_8smallvec', '_7whereat']
    assert all(value == '0x0' and any(crate in symbol for crate in permitted)
               for symbol, value in missing), diagnostic_text
    warnings = re.findall(r'warning: (\d+) functions have mismatched data', diagnostic_text)
    assert sum(map(int, warnings)) == len(missing), diagnostic_text
    residue = re.sub(r"hash-mismatch: No profile record found for '[^']+' with hash = 0x[0-9a-f]+", '', diagnostic_text)
    residue = re.sub(r'warning: \d+ functions have mismatched data', '', residue)
    assert not residue.strip(), diagnostic_text
    (report_output / (name + '.dependency-records.json')).write_text(json.dumps(missing, indent=2) + '\n')
    with target.open('x') as out, (report_output / (name + '.export-diagnostics.log')).open('x') as err:
        subprocess.run([str(llvm / 'llvm-cov'), 'export', '--num-threads=1', str(args.output / name),
                        '--instr-profile=' + str(profile)], stdout=out, stderr=err, check=True)
    assert (report_output / (name + '.export-diagnostics.log')).read_text() == ''.join(
        line + '\n' for line in diagnostic_text.splitlines() if line.startswith('warning:'))
    report['data'].extend(json.loads(target.read_text())['data'])
windows = [
    ('baseline-visible-grid', 'jpeg/writer.rs', '// Non-interleaved scans code the visible component grid', 'let mut block_count:'),
    ('baseline-extra-run-handoff', 'jpeg/writer.rs', 'let mut trailing_zero_runs =', 'block_count += 1;'),
    ('baseline-ac-tokens', 'jpeg/writer.rs', 'fn encode_ac(', 'fn log2_floor_nz('),
]
summary = []
for label, relative, first, last in windows:
    source = Path('zenjxl-decoder/src') / relative
    lines = source.read_text().splitlines()
    begin = next(i + 1 for i, line in enumerate(lines) if first in line)
    end = next(i + 1 for i, line in enumerate(lines) if i + 1 > begin and last in line)
    regions = {}
    for unit in report['data']:
        for function in unit['functions']:
            for region in function['regions']:
                if region[7] != 0 or not function['filenames'][region[5]].endswith(str(source)):
                    continue
                if region[0] < begin or region[2] >= end:
                    continue
                key = tuple(region[:4])
                regions[key] = max(regions.get(key, 0), region[4])
    assert regions, label
    summary.append({'label': label, 'source': str(source), 'first_line': begin, 'last_line': end - 1,
                    'covered': sum(value > 0 for value in regions.values()), 'total': len(regions),
                    'regions': [{'location': key, 'count': value} for key, value in sorted(regions.items())]})
(report_output / 'regions.json').write_text(json.dumps(summary, indent=2) + '\n')
for row in summary:
    print(f'{row["label"]}: {row["covered"]}/{row["total"]} mapped regions executed', flush=True)
