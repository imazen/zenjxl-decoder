#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Replay retained original channel mapping and restore every owned source byte."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
root=Path.cwd();assert(root/'.workongoing').is_file()
files=['frame/mod.rs','jpeg/data.rs','jpeg/jbrd.rs','jpeg/writer.rs','jpeg/mod.rs']
current={};rows=[]
def sha(data):return hashlib.sha256(data).hexdigest()
try:
 for name in files:
  rel=Path('zenjxl-decoder/src')/name;path=root/rel;current[path]=path.read_bytes();source=a.archive/'source'/rel;data=source.read_bytes()
  saved=a.output/'fixed'/rel;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(current[path]);path.write_bytes(data)
  rows.append({'path':str(rel),'fixed_sha256':sha(current[path]),'original_sha256':sha(data)})
 (a.output/'overlay.json').write_text(json.dumps(rows,indent=2)+'\n')
 command=['python3','scripts/jpeg_color_evidence.py','Cargo.toml',str(a.output/'runtime')]
 with(a.output/'replay.log').open('x')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 identity=json.loads((a.output/'runtime/identity.json').read_text());assert identity['exit_code']==101 and result.returncode==101
 text=(a.output/'runtime/test.log').read_text();assert 'test result: FAILED'in text
 for case in ['adobe-one','jfif-over-adobe-zero','custom-ycbcr-ids','ycbcr-ids-adobe-zero']:assert f'{case}: Ok(Some('in text,case
finally:
 for path,data in current.items():path.write_bytes(data);assert path.read_bytes()==data
 (a.output/'restored.json').write_text(json.dumps({'files':len(current),'restored':True},indent=2)+'\n')
print('All four original-mapping failures detected; exact owned source restored.')
