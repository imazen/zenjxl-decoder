#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Extract bound native PNG16 samples using the existing audit PNG reader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'upstream-audit'))
from pngdiff import read_png

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('native_run', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
identity = json.loads((a.native_run / 'identity.json').read_text())
rows = []
for row in identity['rows']:
    source = a.native_run / row['case']
    image = source / 'render.png'
    assert hashlib.sha256(image.read_bytes()).hexdigest() == row['render_sha256']
    assert hashlib.sha256((source / 'reference.jxl').read_bytes()).hexdigest() == row['jxl_sha256']
    w, h, channels, depth, values = read_png(image)
    assert depth == 16 and channels in (1, 3), (row['case'], depth, channels)
    rgb = values if channels == 3 else [v for v in values for _ in range(3)]
    text = f'{w} {h}\n' + '\n'.join(' '.join(map(str, rgb[y*w*3:(y+1)*w*3])) for y in range(h)) + '\n'
    target = a.output / (row['case'] + '.rgb16.txt')
    target.write_text(text)
    rows.append({'case': row['case'], 'jxl_sha256': row['jxl_sha256'], 'render_sha256': row['render_sha256'], 'samples_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(a.output / 'identity.json').write_text(json.dumps(rows, indent=2) + '\n')
print(f'Extracted {len(rows)} independently rendered PNG16 controls.')
