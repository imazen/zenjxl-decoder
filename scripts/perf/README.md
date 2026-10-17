# Decoder performance scripts

Small tools used for the djxl / jxl-rs parity work recorded in `CLAUDE.md`
("Multi-threaded parity work — 2026-10-03"). macOS-oriented: profiles come
from [samply](https://github.com/mstange/samply) and are symbolicated with
`atos`; counters come from `/usr/bin/time -l`.

Build the CLI first (`cargo build --release -p zenjxl-decoder-cli`) and copy
the binary somewhere stable, since each experiment compares two builds.

## Timing

- `ipc.sh <cli> <file>`: one thread, 10 decodes, `f32`. Prints MP/s,
  instructions retired, cycles and IPC. Instruction and cycle counts are
  much less sensitive to background load than wall time; compare cycles.
- `abn.py <threads> <rounds> <file> <label=cmd>...`: interleaved A/B runs,
  median and max MP/s per variant. Environment variables go before the
  command, e.g. `"nopf=JXL_NOPF=1 ./cli --no-cms"`.
- `vsdj.py <cli> <threads> <rounds> <files...>`: interleaved runs of this
  decoder, djxl and upstream jxl-rs, median of per-round means, plus the
  zen/djxl ratio. Set `DJXL` and `JXLRS_CLI` to the binaries. djxl runs
  with `--disable_output`, which decodes to float, so compare with
  `--data-type f32`.
- `lftime.sh <cli> <file>`: median serial LF-phase time at 12 threads, from
  `JXL_PHASE_TIMING=1` (which also prints P2/P3 phase times).

## Profiles

Record with `samply record --save-only -o prof.json.gz -- <cli> ... --speedtest -n 40 <file>`.

- `fnprof.py <prof> <binary> [N]`: self time by innermost own function.
- `fnprof_outer.py <prof> <binary> [N]`: self time by compiled function
  (what a function costs including everything inlined into it).
- `lineprof.py <prof> <binary> [N]`: self time by source line.
- `stackprof.py <prof> <binary> <substring> [N]`: among samples whose stack
  contains `substring`, self time by innermost function (e.g.
  `decode_lf_and_hf_global` to profile only the serial LF stage).

Line and function attribution on NEON code skids: a hot load is often
blamed on the instruction after it. Treat a change as a win only if cycles
drop (`ipc.sh`), not because a profile bucket shrank.

djxl from the libjxl build ships symbols: symbolicate its addresses with
`atos -o djxl -l 0x100000000 <0x1000...>` to compare per-stage shares.

## Checks that found real problems

- **False sharing**: run two processes with N/2 threads each at the same
  time and compare their combined MP/s with one N-thread process. A large
  gap means threads in one process slow each other (here a small
  per-sample buffer shared a cache line, fixed with `util::IsolatedBuf`).
- **Per-group timing**: timestamps around each group's decode, printed per
  thread, show load imbalance and threads started on E-cores (2-3x slower
  groups). Not committed; add an `eprintln!` in the P2 worker loop of
  `frame/render.rs` when needed.
- **Generated code**: `objdump -d --no-show-raw-insn --start-address=...`
  on the hot function; look for bounds-check branch pairs (`b.hi`/`b.ls` to
  panic blocks) and stack reloads (`ldr x, [sp, ...]`) inside the loop.

Test files used: 4 CLIC 2025 photos (~2.8 MP) encoded with libjxl v0.12
`cjxl` (`-d 1 -e 7`, `-d 1 --faster_decoding=4`, `-d 3 -e 7`, lossless
`-d 0 -e 1/3/7`, and `--lossless_jpeg=1` from q90 4:2:0 and 4:4:4 JPEGs).
They are not in the repo.
