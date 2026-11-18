#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Generate explicit JPEG entropy controls and bind unmodified v0.12 results."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('cjxl',type=Path)
p.add_argument('djxl',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
a.output.mkdir(exist_ok=False)
def sha(data):return hashlib.sha256(data).hexdigest()
tools={}
for name in ['cjxl','djxl']:
 path=getattr(a,name).resolve();r=subprocess.run([str(path),'--version'],capture_output=True,check=True);version=(r.stdout+r.stderr).decode();assert 'v0.12.0 a7a9c78' in version,(path,version);tools[name]={'path':str(path),'sha256':sha(path.read_bytes()),'version':version}

def source(layout,long_code,selected,runs,nonzero_position,zero_padding,restart):
 factors={'gray':[0x11],'gray22':[0x22],'444':[0x11]*3,'420':[0x22,0x11,0x11]}[layout]
 width,height=(17,9)if layout=='420'else(16,8)
 blocks_per_mcu=6 if layout=='420'else len(factors)
 total=2*blocks_per_mcu
 out=bytearray([255,0xd8])
 def marker(kind,payload):out.extend(bytes([255,kind])+(len(payload)+2).to_bytes(2,'big')+bytes(payload))
 marker(0xdb,[0]+[1]*64)
 marker(0xc0,[8,*height.to_bytes(2,'big'),*width.to_bytes(2,'big'),len(factors),*sum(([c+1,f,0]for c,f in enumerate(factors)),[])])
 code_length = (10 if long_code is True else long_code) if long_code else 1
 dc=[0]*16;dc[code_length-1]=1
 ac=[0]*16
 if long_code:ac[code_length-1]=4
 else:ac[:4]=[1,1,1,1]
 marker(0xc4,[0,*dc,0,0x10,*ac,0,0xf0,1,0xe1])
 if restart:marker(0xdd,[0,1])
 marker(0xda,[len(factors),*sum(([c+1,0]for c in range(len(factors))),[]),0,63,0])
 bits=[]
 def code(symbol):
  value,length=(symbol,code_length)if long_code else[(0,1),(2,2),(6,3),(14,4)][symbol]
  bits.extend((value>>i)&1 for i in range(length-1,-1,-1))
 def flush():
  for at in range(0,len(bits),8):
   chunk=bits[at:at+8]+[0 if zero_padding else 1]*(8-len(bits[at:at+8]));value=sum(v<<(7-i)for i,v in enumerate(chunk));out.append(value)
   if value==255:out.append(0)
  bits.clear()
 for index in range(total):
  if restart and index==blocks_per_mcu:flush();out.extend([255,0xd0])
  bits.extend([0]*code_length) # category-zero DC.
  position=0
  if index==selected:
   if nonzero_position in ['before','at15']:
    code(2 if nonzero_position=='before' else 3);bits.append(1);position=1 if nonzero_position=='before'else 15
   for _ in range(runs):code(1);position+=16
   if nonzero_position=='after':code(2);bits.append(1);position+=1
  if position<63:code(0)
 flush();out.extend([255,0xd9]);return bytes(out)

cases=[
 ('nonzero-before-zrl','gray',False,0,1,'before',False,False),
 ('zrl-before-nonzero','gray',False,0,1,'after',False,False),
 ('three-zrl','gray',False,0,3,'none',True,False),
 ('zrl-finishes-block','gray',False,0,3,'at15',False,False),
 ('long-codes-nonzero-before','gray',True,1,2,'before',True,True),
 ('long-codes-zrl-before-nonzero','gray',True,0,1,'after',False,True),
 ('gray-absolute22','gray22',False,1,1,'none',False,True),
 ('interleaved-last','444',False,5,2,'before',False,True),
 ('interleaved-first-long','444',True,0,1,'none',True,True),
 ('edge-padding-luma','420',False,7,1,'none',False,True),
 ('edge-visible-chroma-long','420',True,11,2,'before',True,True),
 ('slow11-nonzero-before','gray',11,1,2,'before',True,True),
 ('slow11-zrl-before-nonzero','gray',11,0,1,'after',False,True),
]
rows=[]
for label,*args in cases:
 out=a.output/label;out.mkdir();data=source(*args);(out/'input.jpg').write_bytes(data)
 for name,command in [
  ('encode',[tools['cjxl']['path'],str(out/'input.jpg'),str(out/'reference.jxl'),'--lossless_jpeg=1','--num_threads=1','-e','3']),
  ('render',[tools['djxl']['path'],str(out/'reference.jxl'),str(out/'render.png'),'--bits_per_sample=16','--num_threads=1']),
  ('reconstruct',[tools['djxl']['path'],str(out/'reference.jxl'),str(out/'rebuilt.jpg'),'--num_threads=1'])]:
  r=subprocess.run(command,capture_output=True);(out/(name+'.log')).write_bytes(r.stdout+r.stderr);assert r.returncode==0,(label,name)
 assert(out/'rebuilt.jpg').read_bytes()==data,label
 rows.append({'case':label,'input_sha256':sha(data),'jxl_sha256':sha((out/'reference.jxl').read_bytes()),'reconstructed_sha256':sha((out/'rebuilt.jpg').read_bytes()),'render_sha256':sha((out/'render.png').read_bytes()),'arguments':args})
 (a.output/'results.partial.json').write_text(json.dumps(rows,indent=2))
(a.output/'identity.json').write_text(json.dumps({'tools':tools,'runner_sha256':sha(Path(__file__).read_bytes()),'rows':rows},indent=2))
print('Every explicit native input rendered and reconstructed exactly.')
