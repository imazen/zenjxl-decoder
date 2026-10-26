# zenjxl-decoder dev commands

source-checks:
    ./ci/pull_request_checks.sh

# Standalone interpretation qualification; no encoder/oracle in library builds.
# Export once with git archive, then synchronize only this task's source files.
interpretation-sync export:
    test ! -e "{{export}}/.git"
    test ! -e "{{export}}/.jj"
    test -f "{{export}}/zenjxl-decoder/Cargo.toml"
    cp zenjxl-decoder/src/api/data_types.rs zenjxl-decoder/src/api/decoder.rs "{{export}}/zenjxl-decoder/src/api/"
    cp zenjxl-decoder/src/api/inner/mod.rs "{{export}}/zenjxl-decoder/src/api/inner/"
    cp zenjxl-decoder/src/api/inner/codestream_parser/mod.rs "{{export}}/zenjxl-decoder/src/api/inner/codestream_parser/"
    cp zenjxl-decoder/src/headers/extra_channels.rs zenjxl-decoder/src/headers/frame_header.rs "{{export}}/zenjxl-decoder/src/headers/"
    cp zenjxl-decoder/src/tests/basic_info_profile_guard.rs "{{export}}/zenjxl-decoder/src/tests/"
    mkdir -p "{{export}}/perf/interpretation-validation/src"
    cp perf/interpretation-validation/Cargo.toml perf/interpretation-validation/oracle.c "{{export}}/perf/interpretation-validation/"
    cp perf/interpretation-validation/src/*.rs "{{export}}/perf/interpretation-validation/src/"
    cp justfile "{{export}}/justfile"

stop-reason-sync export:
    test ! -e "{{export}}/.git"
    test ! -e "{{export}}/.jj"
    cp zenjxl-decoder/src/error.rs "{{export}}/zenjxl-decoder/src/error.rs"
    cp zenjxl-decoder/src/api/decoder.rs "{{export}}/zenjxl-decoder/src/api/decoder.rs"
    cp justfile "{{export}}/justfile"

stop-reason-check:
    cargo test --locked -p zenjxl-decoder --no-default-features --lib stop_
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

stop-reason-docs:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked -p zenjxl-decoder --no-deps
    just api-doc

coalescing-sync export:
    just stop-reason-sync "{{export}}"
    cp zenjxl-decoder/src/api/options.rs "{{export}}/zenjxl-decoder/src/api/options.rs"
    cp zenjxl-decoder/src/api/inner/process.rs "{{export}}/zenjxl-decoder/src/api/inner/process.rs"

coalescing-check:
    cargo test --locked -p zenjxl-decoder --no-default-features --lib coalescing_
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

header-budget-sync export:
    just coalescing-sync "{{export}}"
    cp zenjxl-decoder/src/api/inner/codestream_parser/mod.rs zenjxl-decoder/src/api/inner/codestream_parser/non_section.rs "{{export}}/zenjxl-decoder/src/api/inner/codestream_parser/"

header-budget-check filter="frame_header_budget_":
    cargo test --locked -p zenjxl-decoder --no-default-features --lib "{{filter}}"
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

# Native local-plane probe and qualified public raw-layer output.
raw-layer-sync export:
    just header-budget-sync "{{export}}"
    cp zenjxl-decoder/src/api/inner/mod.rs "{{export}}/zenjxl-decoder/src/api/inner/"
    cp zenjxl-decoder/src/api/data_types.rs "{{export}}/zenjxl-decoder/src/api/"
    cp zenjxl-decoder/src/frame/mod.rs zenjxl-decoder/src/frame/decode.rs zenjxl-decoder/src/frame/render.rs "{{export}}/zenjxl-decoder/src/frame/"
    cp zenjxl-decoder/src/tests/mod.rs zenjxl-decoder/src/tests/raw_layers.rs "{{export}}/zenjxl-decoder/src/tests/"

raw-layer-probe artifacts:
    mkdir -p "{{artifacts}}"
    JXL_RAW_LAYER_ARTIFACTS="{{artifacts}}" cargo test --locked -p zenjxl-decoder --no-default-features --lib raw_layer_probe_ -- --nocapture --test-threads=1
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

raw-layer-output-check artifacts:
    mkdir -p "{{artifacts}}"
    JXL_RAW_LAYER_ARTIFACTS="{{artifacts}}" cargo test --locked -p zenjxl-decoder --no-default-features --lib raw_layer_output_ -- --nocapture --test-threads=1
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

raw-layer-output-all artifacts:
    mkdir -p "{{artifacts}}"
    JXL_RAW_LAYER_ARTIFACTS="{{artifacts}}" cargo test --locked -p zenjxl-decoder --all-features --lib raw_layer_output_ -- --nocapture --test-threads=1

raw-layer-regressions:
    cargo test --locked -p zenjxl-decoder --no-default-features --lib coalescing_
    cargo test --locked -p zenjxl-decoder --no-default-features --lib frame_header_budget_
    cargo test --locked -p zenjxl-decoder --no-default-features --lib stop_
    cargo test --locked -p zenjxl-decoder --no-default-features --lib basic_info_profile_guard
    cargo test --locked -p zenjxl-decoder --no-default-features --lib test_scanned_frames_identical_for_decode_and_scan
    cargo test --locked -p zenjxl-decoder --no-default-features --lib test_start_new_frame_

raw-layer-probe-all artifacts:
    mkdir -p "{{artifacts}}"
    JXL_RAW_LAYER_ARTIFACTS="{{artifacts}}" cargo test --locked -p zenjxl-decoder --all-features --lib raw_layer_probe_ -- --nocapture --test-threads=1

raw-layer-oracle captured output python="python3":
    test ! -e "{{output}}"
    mkdir -p "{{output}}"
    cc -std=c11 -Wall -Wextra -Werror perf/raw-layer-validation/oracle.c -ljxl -o "{{output}}/oracle"
    "{{python}}" perf/raw-layer-validation/compare.py "{{captured}}" zenjxl-decoder/resources/test "{{output}}/oracle" "{{output}}"
    "{{python}}" perf/raw-layer-validation/negative.py "{{captured}}" zenjxl-decoder/resources/test "{{output}}/oracle" "{{output}}/negative"

interpretation-resolve:
    cargo metadata --manifest-path perf/interpretation-validation/Cargo.toml --format-version 1 > /dev/null

interpretation-core-check:
    cargo test --locked -p zenjxl-decoder --no-default-features --lib basic_info_profile_guard
    cargo test --locked -p zenjxl-decoder --no-default-features --lib test_scanned_frames_identical_for_decode_and_scan
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

interpretation-header-check:
    cargo test --locked -p zenjxl-decoder --no-default-features --lib headers::frame_header::

interpretation-check artifacts:
    mkdir -p "{{artifacts}}-oracle"
    cc -std=c11 -Wall -Wextra -Werror perf/interpretation-validation/oracle.c -ljxl -o "{{artifacts}}-oracle/jxl-oracle"
    JXL_INTERPRETATION_ARTIFACTS="{{artifacts}}" JXL_INTERPRETATION_ORACLE="{{artifacts}}-oracle/jxl-oracle" cargo test --manifest-path perf/interpretation-validation/Cargo.toml --locked -- --nocapture --test-threads=1
    cargo clippy --manifest-path perf/interpretation-validation/Cargo.toml --locked --all-targets -- -D warnings

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

# Check compatibility after admitting the current archmage/magetypes beta.
beta-cohort-check:
    cargo fmt --check
    cargo clippy --locked --workspace --lib -- -D warnings
    cargo test --locked --workspace --lib --no-fail-fast
