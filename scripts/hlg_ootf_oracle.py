#!/usr/bin/env python3
# Copyright (c) Imazen LLC. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.

"""Build the pinned native HLG component observer and retain its identities."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True,help='new executable path on an executable local filesystem')
p.add_argument('source',type=Path);p.add_argument('build',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
version=subprocess.run([str(a.build/'tools/cjxl'),'--version'],check=True,capture_output=True,text=True)
assert 'v0.12.0 a7a9c78' in version.stdout+version.stderr
header=a.source/'lib/jxl/cms/tone_mapping.h'
assert 'exponent_ < -0.01f || 0.01f < exponent_' in header.read_text()
program=Path(__file__).with_suffix('.cc')
binary=a.binary
assert not binary.exists(), 'preserve existing oracle binaries'
command=['c++','-std=c++17','-O2','-DNDEBUG','-I'+str(a.source),'-I'+str(a.source/'lib/include'),'-I'+str(a.build/'lib/include'),str(program),'-o',str(binary)]
with(a.output/'build.log').open('x') as log: subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
with(a.output/'vectors.tsv').open('x') as out:subprocess.run([str(binary)],check=True,stdout=out)
shutil.copy2(binary,a.output/'oracle')
assert sha(binary)==sha(a.output/'oracle')
files=[program,header,a.source/'lib/jxl/cms/transfer_functions.h',binary,a.output/'vectors.tsv']
(a.output/'identity.json').write_text(json.dumps(dict(command=command,files=[dict(path=str(f),sha256=sha(f))for f in files]),indent=2)+'\n')
