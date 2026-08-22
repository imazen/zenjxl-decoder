#!/usr/bin/env python3
"""Compare two PNGs pixel-by-pixel (stdlib only). Prints size, max abs diff, #diff pixels."""
import sys, zlib, struct

def read_png(path):
    d = open(path, 'rb').read()
    assert d[:8] == b'\x89PNG\r\n\x1a\n', path
    pos = 8; idat = b''; w=h=bd=ct=None
    while pos < len(d):
        ln, = struct.unpack('>I', d[pos:pos+4]); typ = d[pos+4:pos+8]; body = d[pos+8:pos+8+ln]
        if typ == b'IHDR':
            w, h, bd, ct, _, _, il = struct.unpack('>IIBBBBB', body); assert il == 0, 'interlaced'
        elif typ == b'IDAT': idat += body
        elif typ == b'IEND': break
        pos += 12 + ln
    ch = {0:1, 2:3, 4:2, 6:4}[ct]
    bps = bd // 8; bpp = ch * bps; stride = w * bpp
    raw = zlib.decompress(idat); out = bytearray(); prev = bytearray(stride); p = 0
    for _ in range(h):
        f = raw[p]; p += 1; line = bytearray(raw[p:p+stride]); p += stride
        for i in range(stride):
            a = line[i-bpp] if i >= bpp else 0; b = prev[i]; c = prev[i-bpp] if i >= bpp else 0
            if f == 1: line[i] = (line[i] + a) & 255
            elif f == 2: line[i] = (line[i] + b) & 255
            elif f == 3: line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2*c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        out += line; prev = line
    # expand to list of per-pixel tuples of ints (16-bit kept as 16-bit)
    px = []
    if bps == 1:
        vals = list(out)
    else:
        vals = [ (out[i] << 8) | out[i+1] for i in range(0, len(out), 2) ]
    return w, h, ch, bd, vals

def main(a, b):
    wa, ha, ca, bda, va = read_png(a); wb, hb, cb, bdb, vb = read_png(b)
    print(f"A: {wa}x{ha} ch={ca} bd={bda}   B: {wb}x{hb} ch={cb} bd={bdb}")
    if (wa, ha) != (wb, hb): print("SIZE MISMATCH"); return 2
    # compare on common channels; scale to 8-bit if depth differs (report scale note)
    chans = min(ca, cb)
    def get(vals, ch, bd, i, c):
        v = vals[i*ch + c]
        import os
        if os.environ.get('FULL16'): return v if bd == 16 else v << 8
        return v >> 8 if bd == 16 else v  # downscale 16->8 by truncation for comparison
    maxd = [0]*chans; ndiff = 0; hist = {}
    for i in range(wa*ha):
        dp = False
        for c in range(chans):
            d = abs(get(va, ca, bda, i, c) - get(vb, cb, bdb, i, c))
            if d: dp = True; maxd[c] = max(maxd[c], d); hist[d] = hist.get(d, 0) + 1
        if dp: ndiff += 1
    print(f"channels compared={chans} (alpha {'included' if chans in (2,4) else 'not compared'}); differing pixels={ndiff}/{wa*ha}; max abs diff per channel={maxd}")
    if hist: print("diff histogram:", dict(sorted(hist.items())[:12]))
    return 0 if ndiff == 0 else 1

if __name__ == '__main__': sys.exit(main(sys.argv[1], sys.argv[2]))
