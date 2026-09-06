# zenjxl-decoder dev commands

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Profile decode-from-bytes heap allocations with heaptrack (needs heaptrack installed).
# Defaults to the committed bike_web_q85.jxl (2048x2560) decoded 8x; pass a path + iters
# to override. Inspect with: heaptrack_print /tmp/zenjxl-ht.zst
heaptrack-decode *ARGS:
    cargo build -p zenjxl-decoder --release --example heaptrack_decode
    rm -f /tmp/zenjxl-ht.zst
    heaptrack --output /tmp/zenjxl-ht ./target/release/examples/heaptrack_decode {{ARGS}}

# Mirrors .github/workflows/ci.yml: fmt check, clippy with all / no features,
# tests with all features, with none, and with `threads` but no `allow-unsafe`.
# Local CI gate -- run before every push
ci:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo clippy --workspace --all-targets --no-default-features -- -D warnings
    cargo test --release --all --no-fail-fast --all-features
    cargo test --release --all --no-fail-fast --no-default-features
    cargo test --release --all --no-fail-fast --no-default-features --features threads,all-simd

# Env vars in scripts/upstream-audit/speed_compare.sh (THREADS=12 for multi-threaded).
# Speed smoke test of the CLI against an upstream jxl-rs build
speed-compare *ARGS:
    scripts/upstream-audit/speed_compare.sh {{ARGS}}

# Interleaved ARM decode comparisons, full output preserved.
arm-decode-tiers-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --locked -p zenjxl-decoder --bench decode_bench -- --group=decode_tiers --format=llm > "$HOME/tmp/zenjxl-decode-tiers.log" 2>&1

arm-kernel-tiers-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --locked -p zenjxl-decoder --features __bench_kernels --bench kernel_tiers -- --format=llm > "$HOME/tmp/zenjxl-kernel-tiers.log" 2>&1

arm-modular-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --locked -p zenjxl-decoder --bench decode_bench -- --group=modular --format=llm > "$HOME/tmp/zenjxl-modular.log" 2>&1
