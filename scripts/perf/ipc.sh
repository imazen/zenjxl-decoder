#!/bin/bash
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
# usage: ipc.sh <cli> <file> -> MP/s, instructions, cycles, IPC (1 thread, 10 reps)
out=$( /usr/bin/time -l nice -n 5 $1 --no-cms --data-type f32 --num-threads 1 --speedtest -n 10 $2 2>&1 )
mps=$(echo "$out" | grep -o '[0-9.]* MP/s' | tail -1)
ins=$(echo "$out" | grep "instructions retired" | awk '{print $1}')
cyc=$(echo "$out" | grep "cycles elapsed" | awk '{print $1}')
echo "$mps ins=$ins cyc=$cyc ipc=$(echo "scale=2; $ins/$cyc" | bc)"
