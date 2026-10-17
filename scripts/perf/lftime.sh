#!/bin/bash
# Copyright (c) Imazen LLC.
# usage: lftime.sh <cli> <file> -> median LF phase ms (12 threads, 30 decodes)
JXL_PHASE_TIMING=1 nice -n 5 $1 --no-cms --data-type f32 --num-threads 12 --speedtest -n 30 $2 2>&1 | grep -o "lf_par: [0-9.]*ms" | sed 's/lf_par: //; s/ms//' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]}'
