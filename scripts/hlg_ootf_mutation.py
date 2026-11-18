#!/usr/bin/env python3
"""Require native kernel and full-pixel HLG controls to detect the old cutoff."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
root=Path(__file__).resolve().parent.parent;path=root/'zenjxl-decoder/src/color/tf.rs'
fixed=path.read_bytes();source=fixed.decode();assert source.count('if exp.abs() <= 0.01 {')==2
fault=source.replace('if exp.abs() <= 0.01 {','if exp.abs() < 0.1 {').encode()
(a.output/'fixed.rs').write_bytes(fixed);(a.output/'fault.rs').write_bytes(fault)
try:
 path.write_bytes(fault)
 command=[sys.executable,str(root/'scripts/hlg_ootf_evidence.py'),str(root/'Cargo.toml'),str(a.output/'runtime'),'--expect-failure']
 subprocess.run(command,check=True)
 text=(a.output/'runtime/test.log').read_text()
 assert 'complete native HLG row' in text and 'native HLG OOTF row' in text,text
finally:
 path.write_bytes(fixed);assert path.read_bytes()==fixed
 (a.output/'restored.json').write_text(json.dumps(dict(restored=True,sha256=hashlib.sha256(fixed).hexdigest()))+'\n')

(a.output/"verdict.json").write_text(json.dumps(dict(exit_code=0,restored=True))+"\n")
