#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Generate small asymmetric JPEG planes and bind unmodified v0.12 controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('cjxl',type=Path);p.add_argument('djxl',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
def sha(data):return hashlib.sha256(data).hexdigest()
tools={}
for name in ['cjxl','djxl']:
 path=getattr(a,name).resolve();r=subprocess.run([str(path),'--version'],capture_output=True,check=True);version=(r.stdout+r.stderr).decode();assert 'v0.12.0 a7a9c78'in version;tools[name]={'path':str(path),'version':version,'sha256':sha(path.read_bytes())}
def source(ids,adobe,jfif):
 out=bytearray([255,0xd8])
 def marker(kind,data):out.extend(bytes([255,kind])+(len(data)+2).to_bytes(2,'big')+bytes(data))
 if adobe is not None:marker(0xee,(b'Adobe'if adobe!=2 else b'adobe')+bytes([0,100,0,0,0,0,1 if adobe==1 else 0]))
 if jfif:marker(0xe0,b'JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00')
 marker(0xdb,[0]+[1]*64)
 marker(0xc0,[8,0,8,0,16,3,*sum(([i,0x11,0]for i in ids),[])])
 counts=[0]*16;counts[2]=4
 marker(0xc4,[0,*counts,0,1,2,3,0x10,1,*([0]*15),0])
 marker(0xda,[3,*sum(([i,0]for i in ids),[]),0,63,0])
 bits=[]
 for mcu in range(2):
  for category in [1,2,3]:
   cat=category if mcu==0 else 0
   bits.extend((cat>>i)&1 for i in [2,1,0]);bits.extend([1]*cat);bits.append(0)
 while len(bits)%8:bits.append(1)
 for at in range(0,len(bits),8):
  byte=sum(value<<(7-i)for i,value in enumerate(bits[at:at+8]));out.append(byte)
  if byte==255:out.append(0)
 out.extend([255,0xd9]);return bytes(out)
rows=[]
for label,ids,adobe,jfif in [('rgb-ids',b'RGB',None,False),('adobe-zero',b'RGB',0,False),('adobe-one',b'RGB',1,False),('malformed-adobe-rgb-ids',b'RGB',2,False),('default-ids',[1,2,3],None,False),('jfif-over-adobe-zero',b'RGB',0,True),('custom-ycbcr-ids',[4,5,6],None,False),('ycbcr-ids-adobe-zero',[1,2,3],0,False),('custom-rgb-adobe-zero',[4,5,6],0,False)]:
 out=a.output/label;out.mkdir();input=source(ids,adobe,jfif);(out/'input.jpg').write_bytes(input)
 for name,command in [('encode',[tools['cjxl']['path'],str(out/'input.jpg'),str(out/'reference.jxl'),'--lossless_jpeg=1','--num_threads=1','-e','3']),('render',[tools['djxl']['path'],str(out/'reference.jxl'),str(out/'render.png'),'--bits_per_sample=16','--num_threads=1']),('reconstruct',[tools['djxl']['path'],str(out/'reference.jxl'),str(out/'rebuilt.jpg'),'--num_threads=1'])]:
  r=subprocess.run(command,capture_output=True);(out/(name+'.log')).write_bytes(r.stdout+r.stderr);assert r.returncode==0,(label,name,r.stderr)
 assert(out/'rebuilt.jpg').read_bytes()==input
 rows.append({'case':label,'input_sha256':sha(input),'jxl_sha256':sha((out/'reference.jxl').read_bytes()),'reconstructed_sha256':sha((out/'rebuilt.jpg').read_bytes()),'render_sha256':sha((out/'render.png').read_bytes())})
(a.output/'identity.json').write_text(json.dumps({'tools':tools,'runner_sha256':sha(Path(__file__).read_bytes()),'rows':rows},indent=2)+'\n')
print('Every asymmetric native color signal renders and reconstructs exactly.')
