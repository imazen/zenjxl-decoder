#!/usr/bin/env python3
"""Require the no-CMS post-header refusal to detect removal of its guard."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
root=Path(__file__).resolve().parent.parent;path=root/'zenjxl-decoder/src/api/inner/mod.rs'
source=path.read_text();begin=source.index('        // The same restriction is checked while parsing a profile selected')
end=source.index('        self.codestream_parser.output_color_profile = Some(profile);',begin)
fault=source[:begin]+source[end:]
(a.output/'fixed.rs').write_text(source);(a.output/'fault.rs').write_text(fault)
try:
 path.write_text(fault)
 subprocess.run([sys.executable,str(root/'scripts/hlg_ootf_evidence.py'),str(root/'Cargo.toml'),str(a.output/'runtime'),'--expect-failure'],check=True)
 assert 'non-XYB profile change without CMS must refuse' in (a.output/'runtime/test.log').read_text()
finally:
 path.write_text(source);assert path.read_text()==source
 (a.output/'restored.json').write_text(json.dumps(dict(restored=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))+'\n')
(a.output/'verdict.json').write_text(json.dumps(dict(exit_code=0,restored=True))+'\n')
