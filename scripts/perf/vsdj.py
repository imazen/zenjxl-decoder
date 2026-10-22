#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
# usage: vsdj.py <zen-cli> <threads> <rounds> <files...>: interleaved zen/djxl/up runs; median of per-round mean MP/s
import sys, subprocess, re, statistics as st, os
cli, T, R, files = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
DJ = os.environ.get('DJXL', os.path.expanduser('~/work/zen/jxl-encoder/.ci-libjxl/tools/djxl'))
UP = os.environ.get('JXLRS_CLI', os.path.expanduser('~/tmp/jxl-rs/target/release/jxl_cli'))
def run(cmd):
    o = subprocess.run(cmd, capture_output=True, text=True); m = re.findall(r'([0-9.]+) MP/s', o.stdout + o.stderr); return float(m[-1])
res = {'zen': [], 'dj': [], 'up': []}
for r in range(R):
    z = st.mean(run(['nice','-n','5',cli,'--no-cms','--data-type','f32','--num-threads',str(T),'--speedtest','-n','11','--warmup-reps','1',f]) for f in files)
    d = st.mean(run(['nice','-n','5',DJ,f,'--num_reps=11','--num_threads=%d' % (0 if T == 1 else T),'--disable_output']) for f in files)
    u = st.mean(run(['nice','-n','5',UP,'--data-type','f32','--num-threads',str(T),'--speedtest','-n','11','--warmup-reps','1',f]) for f in files)
    res['zen'].append(z); res['dj'].append(d); res['up'].append(u)
print(' '.join(f"{k} {st.median(v):6.1f}" for k, v in res.items()), ' zen/dj %.3f' % (st.median(res['zen'])/st.median(res['dj'])))
