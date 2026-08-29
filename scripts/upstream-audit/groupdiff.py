# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pngdiff import read_png
a, b, G = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 256
wa, ha, ca, bda, va = read_png(a); wb, hb, cb, bdb, vb = read_png(b)
assert (wa, ha, ca) == (wb, hb, cb)
gx, gy = (wa + G - 1)//G, (ha + G - 1)//G
bad = [[0]*gx for _ in range(gy)]
for y in range(ha):
    for x in range(wa):
        i = (y*wa + x)*ca
        d = max(abs(va[i+c]-vb[i+c]) for c in range(min(ca,3)))
        if d > 1: bad[y//G][x//G] = max(bad[y//G][x//G], d)
print(f"groups {gx}x{gy}; groups with any pixel diff>1 (max diff shown):")
for row in bad: print(' '.join(f"{v:5d}" if v else '    .' for v in row))
print("differing groups:", sum(1 for r in bad for v in r if v), "/", gx*gy)
