#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Replay the original grid choice in the owned checkout, preserving the original binary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('revision')
p.add_argument('output',type=Path)
a=p.parse_args()
a.output.mkdir(exist_ok=False)
source=Path('zenjxl-decoder/src/frame/modular/transforms/mod.rs')
fixed=source.read_bytes()
before=subprocess.check_output(['jj','file','show','-r',a.revision,str(source)])
assert before!=fixed
(a.output/'before.rs').write_bytes(before)
(a.output/'after.rs').write_bytes(fixed)
identity=dict(revision=a.revision,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(fixed).hexdigest())
env=os.environ.copy();env['CARGO_TERM_COLOR']='never'
try:
    source.write_bytes(before)
    for name in ['global_squeeze_inputs_survive_a_larger_intermediate_channel','transposed_global_squeeze_inputs_preserve_every_source_code']:
        path=a.output/(name+'.log')
        with path.open('x') as log:
            result=subprocess.run(['cargo','test','--locked','-p','zenjxl-decoder','--test','global_squeeze_grid','--',name,'--exact','--nocapture','--test-threads=1'],stdout=log,stderr=subprocess.STDOUT,env=env)
        log=path.read_text()
        assert result.returncode!=0 and 'row < self.num_rows' in log,(name,result.returncode)
        executable=Path(re.search(r'Running tests/global_squeeze_grid.rs \(([^)]+)\)',log).group(1))
        binary=a.output/'before-binary'
        if not binary.exists():shutil.copy2(executable,binary)
        digest=hashlib.sha256(executable.read_bytes()).hexdigest()
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==digest
        identity['binary_sha256']=digest
        print(name+': original grid choice detected',flush=True)
finally:
    source.write_bytes(fixed)
    assert source.read_bytes()==fixed
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2))
    print('restored fixed source exactly',flush=True)
