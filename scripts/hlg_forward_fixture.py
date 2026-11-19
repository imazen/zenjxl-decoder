#!/usr/bin/env python3
# Copyright (c) Imazen LLC. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.

"""Retain an unmodified v0.12 producer and its independent linear-float decode."""
import argparse,hashlib,json,math,struct,subprocess,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('build',type=Path);p.add_argument('output',type=Path);p.add_argument('binary',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False);assert not a.binary.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
version=subprocess.run([str(a.build/'tools/djxl'),'--version'],capture_output=True,text=True,check=True)
assert 'v0.12.0 a7a9c78' in version.stdout+version.stderr
program=Path(__file__).with_suffix('.cc')
command=['c++','-std=c++17','-O2','-I'+str(a.source/'lib/include'),'-I'+str(a.build/'lib/include'),str(program),'-L'+str(a.build/'lib'),'-Wl,-rpath,'+str(a.build/'lib'),'-ljxl','-o',str(a.binary)]
with(a.output/'build.log').open('x')as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
subprocess.run([str(a.binary),str(a.output/'input.jxl')],check=True)
with(a.output/'native.log').open('x')as log:subprocess.run([str(a.build/'tools/djxl'),str(a.output/'input.jxl'),str(a.output/'native.pfm'),'--num_threads=1','--color_space=RGB_D65_SRG_Rel_Lin'],check=True,stdout=log,stderr=subprocess.STDOUT)
header,dimensions,scale,raw=(a.output/'native.pfm').read_bytes().split(b'\n',3)
assert header==b'PF'and dimensions==b'9 3';values=struct.unpack(('>'if float(scale)>0 else '<')+'81f',raw)
assert all(math.isfinite(v)for v in values)
assert all(values[i]==values[i%3]for i in range(81))
bits=struct.unpack('<3I',struct.pack('<3f',*values[:3]))
(a.output/'expected.bits').write_text(' '.join('%08x'%v for v in bits)+'\n')
shutil.copy2(a.binary,a.output/'producer')
files=[program,a.binary,a.output/'input.jxl',a.output/'native.pfm',a.output/'expected.bits']
(a.output/'identity.json').write_text(json.dumps(dict(exit_code=0,command=command,files=[dict(path=str(f),sha256=sha(f))for f in files],native_version=version.stdout+version.stderr,expected_linear=list(values[:3])),indent=2)+'\n')
