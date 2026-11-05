#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
# usage: lineprof.py <profile.json.gz> <binary> [N]
# Leaf-sample self time by innermost source line (atos -i), for the given binary.
import gzip, json, sys, subprocess, collections
prof, binary = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
p = json.load(gzip.open(prof))
libs = p['libs']
binname = binary.split('/')[-1]
counts = collections.Counter(); total = 0
for t in p['threads']:
    ft, st, fn, rt = t['frameTable'], t['stackTable'], t['funcTable'], t['resourceTable']
    for s in t['samples']['stack']:
        if s is None: continue
        total += 1
        f = st['frame'][s]
        func = ft['func'][f]
        res = fn['resource'][func]
        lib = rt['lib'][res] if res is not None and res >= 0 else None
        if lib is None or libs[lib]['name'] != binname: 
            counts[('other', libs[lib]['name'] if lib is not None else '?')] += 1
            continue
        counts[('addr', ft['address'][f])] += 1
addrs = [k[1] for k, v in counts.items() if k[0] == 'addr']
lines = {}
for i in range(0, len(addrs), 500):
    chunk = addrs[i:i+500]
    out = subprocess.run(['atos', '-o', binary, '-l', '0x100000000', '-i'] + [hex(0x100000000 + a) for a in chunk], capture_output=True, text=True).stdout
    # with -i, each address may print multiple lines (inline chain), separated by blank line
    blocks = out.strip('\n').split('\n\n')
    for a, b in zip(chunk, blocks):
        chain = b.strip().split('\n')
        own = [l for l in chain if '/rustlib/' not in l and '.cargo/registry' not in l and ('.rs:' in l)]
        lines[a] = own[0] if own else chain[0]
agg = collections.Counter(); aggfn = collections.Counter()
for k, v in counts.items():
    if k[0] == 'addr':
        l = lines.get(k[1], '?')
        loc = l[l.rfind('(')+1:-1] if '(' in l else l
        func = l.split(' (in ')[0]
        agg[loc] += v
        aggfn[func[:90]] += v
    else:
        agg['[' + k[1] + ']'] += v
print(f"total samples {total}")
for loc, v in agg.most_common(N):
    print(f"{100*v/total:5.1f}%  {loc}")
