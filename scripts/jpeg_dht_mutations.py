#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Check five JPEG-DHT faults in the owned tree, restoring each source exactly."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
root = Path('zenjxl-decoder/src')
cases = [
    ('reject-empty', root / 'jpeg/jbrd.rs',
     'if !empty_segment {',
     'if empty_segment { return Err(Error::InvalidJbrd("huffman table has no symbols".into())); }\n        if !empty_segment {',
     ['--test', 'jpeg_dht_contract'], 'native_empty_dht_and_zero_symbol_table_reconstruct_exactly'),
    ('write-empty-as-table', root / 'jpeg/writer.rs',
     'if hc.empty_segment {', 'if false && hc.empty_segment {',
     ['--test', 'jpeg_dht_contract'], 'native_empty_dht_and_zero_symbol_table_reconstruct_exactly'),
    ('drop-reconstruction-error', root / 'api/inner/codestream_parser/sections.rs',
     'Err(error) => self.jpeg_recon_error = Some(error.decompose().0),',
     'Err(_error) => {},',
     ['--test', 'jpeg_dht_contract'], 'malformed_jbrd_is_an_error_without_changing_pixel_decode'),
    ('drop-take-error', root / 'api/inner/mod.rs',
     'if let Some(error) = self.codestream_parser.jpeg_recon_error.take() {',
     'if let Some(error) = None::<crate::error::Error> {',
     ['--lib'], 'tests::jpeg_reconstruction::malformed_native_jbrd_error_survives_unchecked_take'),
    ('allow-mixed-empty', root / 'jpeg/writer.rs',
     'if *idx != start || !hc.is_last {', 'if false && (*idx != start || !hc.is_last) {',
     ['--lib'], 'jpeg::jbrd::fallible_alloc_tests::native_dht_empty_record_must_occupy_its_own_segment'),
]
for name, source, old, new, target, test in cases:
    output = args.output / name
    output.mkdir()
    fixed = source.read_bytes()
    text = fixed.decode()
    assert text.count(old) == 1, name
    mutated = text.replace(old, new).encode()
    (output / 'fixed.rs').write_bytes(fixed)
    (output / 'fault.rs').write_bytes(mutated)
    record = {'source': str(source), 'fixed_sha256': hashlib.sha256(fixed).hexdigest(),
              'fault_sha256': hashlib.sha256(mutated).hexdigest()}
    try:
        source.write_bytes(mutated)
        command = ['cargo', 'test', '--locked', '--release', '-p', 'zenjxl-decoder',
                   '--features', 'jpeg', *target, '--no-run', '--message-format=json']
        with (output / 'build.jsonl').open('x') as log:
            subprocess.run(command, stdout=log, check=True)
        rows = [json.loads(line) for line in (output / 'build.jsonl').read_text().splitlines()]
        binaries = [row['executable'] for row in rows if row.get('reason') == 'compiler-artifact'
                    and row.get('executable') and row['target']['name'] == ('jpeg_dht_contract' if '--test' in target else 'zenjxl_decoder')]
        assert len(binaries) == 1, binaries
        binary = Path(binaries[0])
        shutil.copy2(binary, output / 'test-binary')
        record['binary_sha256'] = hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest()
        run = [str(binary), test, '--exact', '--nocapture', '--test-threads=1']
        with (output / 'test.log').open('x') as log:
            result = subprocess.run(run, stdout=log, stderr=subprocess.STDOUT)
        log = (output / 'test.log').read_text()
        assert result.returncode != 0 and re.search(r'test result: FAILED\. 0 passed; 1 failed;', log), (name, result.returncode, log)
        record.update(command=run, exit_code=result.returncode, detected=True)
        print(f'{name}: fault detected by {test}', flush=True)
    finally:
        source.write_bytes(fixed)
        assert source.read_bytes() == fixed
        (output / 'identity.json').write_text(json.dumps(record, indent=2) + '\n')
