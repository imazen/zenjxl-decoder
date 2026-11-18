#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Freeze bound native JPEG color streams and complete PNG16 sample codes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'upstream-audit'))
from pngdiff import read_png
p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);a=p.parse_args()
identity=json.loads((a.native/'identity.json').read_text());fixtures=Path('zenjxl-decoder/tests/testdata')
def sha(data):return hashlib.sha256(data).hexdigest()
for row in identity['rows']:
 label=row['case'];root=a.native/label
 for filename,suffix,key in [('input.jpg','jpg.hex','input_sha256'),('reference.jxl','jxl.hex','jxl_sha256')]:
  data=(root/filename).read_bytes();assert sha(data)==row[key]
  target=fixtures/f'jpeg_scan_v012_color-{label}.{suffix}';text=data.hex()+'\n'
  if target.exists():assert target.read_text()==text
  else:target.write_text(text)
 assert sha((root/'render.png').read_bytes())==row['render_sha256']
 width,height,channels,depth,values=read_png(root/'render.png');assert(width,height,channels,depth)==(16,8,3,16)
 target=fixtures/f'jpeg_scan_v012_color-{label}.rgb16.txt';text='16 8\n'+' '.join(map(str,values))+'\n'
 if target.exists():assert target.read_text()==text
 else:target.write_text(text)
print('Exact native source, codestream and full PNG16 text controls bound.')
