#!/usr/bin/env python3
# Copyright (c) Imazen LLC. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.

"""Report exact HLG kernel regions from checksum-retained passing executables."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm',type=Path);p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
p.add_argument('--package',choices=['jxl','zenjxl-decoder'],default='zenjxl-decoder')
p.add_argument('--kind',choices=['hlg','profile','gray'],default='hlg')
a=p.parse_args();a.output.mkdir(exist_ok=False)
identity=json.loads((a.archive/'identity.json').read_text());assert identity['exit_code']==0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.archive/'test-binary')==identity['binary_sha256']
assert sha(a.archive/'test.profraw')==identity['profile_sha256']
for row in identity['sources']:assert sha(a.archive/'source'/row['path'])==row['sha256'],row['path']
profile=a.output/'merged.profdata'
subprocess.run([str(a.llvm/'llvm-profdata'),'merge','-sparse',str(a.archive/'test.profraw'),'-o',str(profile)],check=True)
with(a.output/'coverage.json').open('x')as out,(a.output/'diagnostics.log').open('x')as err:
 subprocess.run([str(a.llvm/'llvm-cov'),'export',str(a.archive/'test-binary'),'--instr-profile='+str(profile)],stdout=out,stderr=err,check=True)
report=json.loads((a.output/'coverage.json').read_text())
relative=a.package+'/src/color/tf.rs'
names=['hlg_ootf_inner_vec']if a.package=='jxl'else['hlg_ootf_inner_precise','hlg_ootf_inner','hlg_scene_to_display','hlg_display_to_scene']
if a.kind=='profile':relative='zenjxl-decoder/src/api/inner/mod.rs';names=['set_output_color_profile']
if a.kind=='gray':relative='jxl/src/frame/modular/transforms/meta_apply.rs';names=['squeeze_grid_propagation']
lines=(a.archive/'source'/relative).read_text().splitlines()
rows=[]
for name in names:
 if a.kind=='gray':
  start=next(i for i,line in enumerate(lines)if '// Grid placement belongs to the coded input channels.' in line)
  end=next(i for i in range(start,len(lines))if '.max(buffer_info[buf_in[1]].grid_kind);' in lines[i])
 else:
  starts=[i for i,line in enumerate(lines)if re.search(r'\bfn '+name+r'(?:<[^>]*>)?\(',line)]
  assert starts,name
  start=starts[0];indent=lines[start][:len(lines[start])-len(lines[start].lstrip())]
  end=next(i for i in range(start+1,len(lines))if lines[i]==indent+'}')
 counts={};instances=[]
 for unit in report['data']:
  for fun in unit['functions']:
   contributed=False
   for reg in fun['regions']:
    if reg[7]!=0 or not fun['filenames'][reg[5]].endswith(relative):continue
    if reg[0]<start+1 or reg[2]>end+1:continue
    key=tuple(reg[:4]);counts[key]=max(counts.get(key,0),reg[4]);contributed |= reg[4]>0
   if contributed:instances.append(fun['name'])
 assert counts,(relative,name)
 rows.append(dict(function=name,start=start+1,end=end+1,body_sha256=hashlib.sha256('\n'.join(lines[start:end+1]).encode()).hexdigest(),covered=sum(v>0 for v in counts.values()),total=len(counts),missing=[list(k)for k,v in counts.items()if v==0],observed_instances=sorted(set(instances))))
(a.output/'regions.json').write_text(json.dumps(dict(binary_sha256=identity['binary_sha256'],profile_sha256=identity['profile_sha256'],functions=rows,note='Instrumented source-region execution; not input-space, all-platform or all-color coverage.'),indent=2)+'\n')
