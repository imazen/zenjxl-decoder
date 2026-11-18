#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Bind runtime failures for restored baseline scan faults, restoring owned source."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
source=Path('zenjxl-decoder/src/jpeg/writer.rs')
fixed=source.read_bytes();text=fixed.decode()
def digest(data):return hashlib.sha256(data).hexdigest()
def replace_once(old,new):
 assert text.count(old)==1,(old,text.count(old));return text.replace(old,new,1)
loop='    for _ in 0..trailing_zero_runs {\n        bw.write_huffman(table, 0xF0);\n    }\n'
assert text.count(loop)==1
moved=text.replace(loop,'',1);before='    for zi in 1..=last_nonzero_zi {';assert moved.count(before)==1;moved=moved.replace(before,loop+'\n'+before,1)
old=subprocess.check_output(['git','show','f09a223208cfb3c7d0cd5856307653d53d19fdcf:zenjxl-decoder/src/jpeg/writer.rs'],text=True)
lo=old.index('        // Calculate MCU dimensions\n');hi=old.index('        let mut block_count:',lo)
a0=text.index('        // Non-interleaved scans code the visible component grid');a1=text.index('        let mut block_count:',a0)
cases=[
 ('drop-runs',replace_once('    for _ in 0..trailing_zero_runs {','    for _ in 0..0 {'),False,'native_baseline_and_progressive_scan_signals_reconstruct_exactly'),
 ('before-nonzero',moved,False,'native_nonzero_restart_edge_and_long_code_tokens_reconstruct_exactly'),
 ('old-grid',text[:a0]+old[lo:hi]+text[a1:],False,'native_nonzero_restart_edge_and_long_code_tokens_reconstruct_exactly'),
 ('extra-eob',replace_once('    if trailing_end < 63 {','    if true {'),False,'native_nonzero_restart_edge_and_long_code_tokens_reconstruct_exactly'),
 ('allow-extent',replace_once('.filter(|&end| end <= 63)','.filter(|&_end| true)'),True,'baseline_trailing_zrl_bits_and_extent_errors'),
]
records=[]
for name,mutated,is_lib,selection in cases:
 out=a.output/name;out.mkdir();mutation=mutated.encode();(out/'fixed.rs').write_bytes(fixed);(out/'fault.rs').write_bytes(mutation)
 try:
  assert source.read_bytes()==fixed;source.write_bytes(mutation)
  command=['cargo','test','--locked','-p','zenjxl-decoder','--features','jpeg',*(['--lib']if is_lib else['--test','jpeg_scan_contract']),'--no-run','--message-format=json']
  with(out/'build.jsonl').open('x')as stdout,(out/'build.log').open('x')as stderr:subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
  artifacts=[json.loads(l)for l in(out/'build.jsonl').read_text().splitlines()];name_target='zenjxl_decoder'if is_lib else'jpeg_scan_contract';binaries=[Path(r['executable'])for r in artifacts if r.get('reason')=='compiler-artifact'and r.get('executable')and r['target']['name']==name_target];assert len(binaries)==1
  binary=binaries[0];shutil.copy2(binary,out/'test-binary')
  with(out/'test.log').open('x')as log:r=subprocess.run([str(binary),selection,'--nocapture','--test-threads=1'],stdout=log,stderr=subprocess.STDOUT)
  log=(out/'test.log').read_text();assert r.returncode!=0 and re.search(r'test result: FAILED\. 0 passed; 1 failed;',log),(name,r.returncode)
  record={'case':name,'exit_code':r.returncode,'fixed_sha256':digest(fixed),'fault_sha256':digest(mutation),'binary_sha256':digest((out/'test-binary').read_bytes()),'command':command,'selection':selection}
  (out/'identity.json').write_text(json.dumps(record,indent=2));records.append(record);print(name+': actual runtime regression detects restored fault',flush=True)
 finally:
  assert source.read_bytes()==mutation;source.write_bytes(fixed);assert source.read_bytes()==fixed
 (a.output/'faults.json').write_text(json.dumps(records,indent=2))
