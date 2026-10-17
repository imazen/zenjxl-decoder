#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
# usage: abn.py <threads> <rounds> <file> <label=cmd>...  interleaved; median/max MP/s per variant
import sys, subprocess, re, statistics as st, os
T, R, f = sys.argv[1], int(sys.argv[2]), sys.argv[3]
vs = [a.split('=', 1) for a in sys.argv[4:]]
res = {l: [] for l, _ in vs}
for r in range(R):
    for lab, cmd in vs:
        out = subprocess.run(f"nice -n 5 env {cmd} --data-type f32 --num-threads {T} --speedtest -n 11 --warmup-reps 1 {f}", shell=True, capture_output=True, text=True)
        m = re.findall(r'([0-9.]+) MP/s', out.stdout + out.stderr)
        res[lab].append(float(m[-1]))
for lab, _ in vs:
    v = res[lab]; print(f"{lab:10s} med {st.median(v):6.1f} max {max(v):6.1f}  {' '.join('%.0f'%x for x in v)}")
