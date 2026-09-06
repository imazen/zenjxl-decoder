# Modular profile captures

Production `17df9a31`, Apple M4 Pro, Rust 1.98 / LLVM 22. Built using `cargo build --locked --release -p zenjxl-decoder --example heaptrack_decode`. No target-cpu=native. Run under nice -n19 with four Rayon/OMP/build threads.

- `/Users/lilith/tmp/arm-all-2026-09-06/jxl-modular-sample.txt` — SHA-256 `7d7f3da2d3a5259633f1927f2bab20dc8769ef129926d55efd5d2a30c09b08bd`
- `/Users/lilith/tmp/arm-all-2026-09-06/jxl-modular-profile-run.log` — SHA-256 `361030640cd6a443395be0f96e82cbe95c9950bd98a477b1f5a569c7b7e9ec40`
- `/Users/lilith/tmp/arm-all-2026-09-06/jxl-wp-before.s` — SHA-256 `3456652af7ea79a200a2c596734b847158559a9ae30a4d4fe04a4bf550e48c6c`

Command: `target/release/examples/heaptrack_decode zenjxl-decoder/resources/test/green_queen_modular_e3.jxl 2000`, with `/usr/bin/time -l`. Sampling: `sample <pid> 5 -file <sample path>`. No cloud/NAS mirror made.
