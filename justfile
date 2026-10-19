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
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench -p zenjxl-decoder --bench decode_bench -- --group=decode_tiers --cpu-policy=unrestricted --format=llm > "$HOME/tmp/zenjxl-decode-tiers.log" 2>&1

arm-kernel-tiers-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --locked -p zenjxl-decoder --features __bench_kernels --bench kernel_tiers -- --format=llm > "$HOME/tmp/zenjxl-kernel-tiers.log" 2>&1

arm-modular-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --locked -p zenjxl-decoder --bench decode_bench -- --group=modular --format=llm > "$HOME/tmp/zenjxl-modular.log" 2>&1

# JPEG reconstruction regression gate, including standalone terminal restart markers.
jpeg-reconstruction-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 RUST_TEST_THREADS=4
    mkdir -p "$HOME/tmp/zenjxl-decoder"
    nice -n 19 cargo test --locked --release -p zenjxl-decoder --features jpeg --lib > "$HOME/tmp/zenjxl-decoder/jpeg-{{label}}.log" 2>&1
    rg 'test result:' "$HOME/tmp/zenjxl-decoder/jpeg-{{label}}.log"

# Reproduce both WASI CI configurations with explicit corpus policy and full logs.
wasm-ci-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    export ZENJXL_ALLOW_MISSING_CORPUS=1
    mkdir -p "$HOME/tmp/zenjxl-decoder"
    for features in none wasm128; do
        args=()
        if [[ "$features" != none ]]; then args+=(--features "$features"); fi
        nice -n 19 cargo test --locked --release -p zenjxl-decoder --no-fail-fast --no-default-features "${args[@]}" --target wasm32-wasip1 > "$HOME/tmp/zenjxl-decoder/wasm-{{label}}-$features.log" 2>&1
        rg 'test result:' "$HOME/tmp/zenjxl-decoder/wasm-{{label}}-$features.log"
    done

# Native Linux with multilib, matching CI's i686 test selection. Invoke through
# run-heavy on shared Linux hosts to bound the build's memory and CPU use.
i686-ci-check label:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
    export ZENJXL_ALLOW_MISSING_CORPUS=1
    mkdir -p "$HOME/tmp/zenjxl-decoder"
    for features in none all; do
        args=(--no-default-features)
        if [[ "$features" == all ]]; then args=(--all-features); fi
        nice -n 19 cargo test --locked --release --workspace --no-fail-fast "${args[@]}" --target i686-unknown-linux-gnu -- --test-threads=1 > "$HOME/tmp/zenjxl-decoder/i686-{{label}}-$features.log" 2>&1
        rg 'test result:' "$HOME/tmp/zenjxl-decoder/i686-{{label}}-$features.log"
    done

# Reproduce configured-memory and post-codestream parser regressions.
inventory-behavior-check:
    cargo test --locked -p zenjxl-decoder --lib hf_table_budget_tests
    cargo test --locked -p zenjxl-decoder --lib test_box_errors_after_the_codestream
    cargo test --locked -p zenjxl-decoder --lib test_junk_after_the_codestream
    cargo test --locked -p zenjxl-decoder --lib test_codestream_length_after_a_seek
    cargo test --locked -p zenjxl-decoder --lib test_scan_needs_no_frame_memory

# Explicit fixtures; no missing-file skips. Supply only independently valid cases.
icc-readback +files:
    cargo run --locked -p zenjxl-decoder --example icc_readback -- {{files}}

layout-check:
    cargo test --locked -p zenjxl-decoder --lib test_codestream_layout
    cargo test --locked -p zenjxl-decoder --lib failed_or_nested_parse
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

scan-profile-build:
    cargo build --locked -p zenjxl-decoder --example header_scan --example icc_readback

layout-memory-check:
    cargo test --locked -p zenjxl-decoder --lib layout_
    cargo test --locked -p zenjxl-decoder --lib failed_or_nested_parse
    cargo test --locked -p zenjxl-decoder --lib failed_log_reservation
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

icc-valid-check list out:
    #!/usr/bin/env bash
    set -euo pipefail
    mapfile -t cases < {{quote(list)}}
    target/debug/examples/icc_readback --write-icc {{quote(out)}} "${cases[@]}"

# Keep the expected malformed-input status while retaining the heap trace.
scan-heap-profile input output:
    #!/usr/bin/env bash
    set -euo pipefail
    result=0
    heaptrack -o {{quote(output)}} target/debug/examples/header_scan {{quote(input)}} || result=$?
    test "$result" -eq 1

icc-oracle list native out djxl:
    just icc-valid-check {{quote(list)}} {{quote(native)}}
    python3 scripts/check_icc_oracle.py {{quote(list)}} {{quote(native)}} {{quote(out)}} {{quote(djxl)}}

# Check compatibility after admitting the current archmage/magetypes beta.
beta-cohort-check:
    cargo fmt --check
    cargo clippy --locked --workspace --lib -- -D warnings
    cargo test --locked --workspace --lib --no-fail-fast

# Header/layout changes: full native suites and standalone fuzz compilation.
inventory-final-check:
    cargo fmt --all -- --check
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
    cargo test --locked --workspace --all-features --no-fail-fast
    cargo test --locked --workspace --no-default-features --no-fail-fast
    cargo check --manifest-path fuzz/Cargo.toml --bin layout_scan --bin container_boxes

# Call through run-heavy; artifacts and accumulated inputs stay in block storage.
layout-fuzz seconds="600":
    cargo +nightly fuzz run --target x86_64-unknown-linux-gnu layout_scan /mnt/v/fuzzes/zenjxl-decoder/corpus/layout_scan -- -max_total_time={{seconds}} -max_len=65536 -rss_limit_mb=16384 -timeout=30 -artifact_prefix=/mnt/v/fuzzes/zenjxl-decoder/artifacts/

large-icc-layout-check:
    cargo test --locked -p zenjxl-decoder --lib test_large_compressible_icc_one_shot_and_fragmented
    cargo clippy --locked -p zenjxl-decoder --lib --tests --all-features -- -D warnings

# Match CI's optimized minimal-feature matrix; scalar decode sweeps are expensive in debug.
minimal-release-check:
    cargo test --release --locked --workspace --no-default-features --no-fail-fast

