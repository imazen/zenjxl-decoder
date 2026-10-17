#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
# usage: stackprof.py <profile.json.gz> <binary> <stack-substring> [N]
# Among samples whose (symbolicated) stack contains <stack-substring>, self time by innermost own function.
import gzip, json, sys, subprocess, collections
prof, binary, needle = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 30
p = json.load(gzip.open(prof)); libs = p['libs']; binname = binary.split('/')[-1]
addrs = set(); samples = []
for t in p['threads']:
    ft, st, fn, rt = t['frameTable'], t['stackTable'], t['funcTable'], t['resourceTable']
    for s in t['samples']['stack']:
        if s is None: continue
        chain = []
        while s is not None:
            f = st['frame'][s]; res = fn['resource'][ft['func'][f]]
            lib = rt['lib'][res] if res is not None and res >= 0 else None
            chain.append(ft['address'][f] if lib is not None and libs[lib]['name'] == binname else None)
            s = st['prefix'][s]
        samples.append(chain); addrs.update(a for a in chain if a is not None)
addrs = list(addrs); sym = {}
for i in range(0, len(addrs), 500):
    ch = addrs[i:i+500]
    out = subprocess.run(['atos','-o',binary,'-l','0x100000000','-i']+[hex(0x100000000+a) for a in ch], capture_output=True, text=True).stdout.strip('\n').split('\n\n')
    for a, b in zip(ch, out): sym[a] = b
tot = 0; c = collections.Counter()
for chain in samples:
    txt = '\n'.join(sym.get(a, '') for a in chain if a is not None)
    if needle not in txt: continue
    tot += 1
    leaf = chain[0]
    if leaf is None: c['[other lib]'] += 1; continue
    l = sym[leaf].split('\n')[0].split(' (in ')[0]
    c[l[-90:]] += 1
print('matching samples', tot, 'of', len(samples))
for k, v in c.most_common(N): print('%5.1f%% %s' % (100*v/max(tot,1), k))
