#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
# usage: fnprof_outer.py <profile.json.gz> <binary> [N]
# Self time by the compiled (outermost inlined-into) zenjxl function of each sample.
import gzip, json, sys, subprocess, collections, re
prof, binary = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
OUTER = True
p = json.load(gzip.open(prof))
binname = binary.split('/')[-1]
libs = p['libs']
counts = collections.Counter(); total = 0; other = collections.Counter()
for t in p['threads']:
    ft, st, fn, rt = t['frameTable'], t['stackTable'], t['funcTable'], t['resourceTable']
    for s in t['samples']['stack']:
        if s is None: continue
        total += 1
        f = st['frame'][s]; res = fn['resource'][ft['func'][f]]
        lib = rt['lib'][res] if res is not None and res >= 0 else None
        if lib is None or libs[lib]['name'] != binname:
            other[libs[lib]['name'] if lib is not None else '?'] += 1; continue
        counts[ft['address'][f]] += 1
addrs = list(counts)
names = {}
for i in range(0, len(addrs), 400):
    chunk = addrs[i:i+400]
    out = subprocess.run(['atos','-o',binary,'-l','0x100000000','-i']+[hex(0x100000000+a) for a in chunk], capture_output=True, text=True).stdout.strip('\n').split('\n\n')
    for a, b in zip(chunk, out):
        chain = b.split('\n')
        pick = None
        for l in (reversed(chain) if OUTER else chain):
            sym = l.split(' (in ')[0]
            iz = min([i for i in (sym.find('14zenjxl_decoder'), sym.find('19zenjxl_decoder_simd')) if i >= 0], default=-1)
            if iz < 0: continue
            ic = min([i for i in (sym.find('4core'), sym.find('5alloc'), sym.find('3std')) if i >= 0], default=10**9)
            if iz < ic:
                pick = sym; break
        if pick is None: pick = chain[-1].split(' (in ')[0]
        # demangle-ish: keep path components
        # v0 mangling: decode <len><ident> runs, skipping crate hashes
        ids = []; i = 0
        while i < len(pick):
            m = re.match(r'(\d+)', pick[i:])
            if m:
                n = int(m.group(1)); i += len(m.group(1))
                ident = pick[i:i+n]; i += n
                if ident and not ident.startswith('_') and len(ident) > 1: ids.append(ident)
            else:
                i += 1
        keep = [x for x in ids if x not in ('zenjxl_decoder','zenjxl_decoder_simd','aarch64','neon','NeonDescriptor','core','alloc')]
        names[a] = '::'.join(keep[-4:])[-90:]
agg = collections.Counter()
for a, v in counts.items(): agg[names[a]] += v
for k, v in other.items(): agg['['+k+']'] += v
print(f"total {total}")
for k, v in agg.most_common(N): print(f"{100*v/total:5.1f}%  {k}")
