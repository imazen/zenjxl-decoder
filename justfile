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

# Check compatibility after admitting the current archmage/magetypes beta.
beta-cohort-check:
    cargo fmt --check
    cargo clippy --locked --workspace --lib -- -D warnings
    cargo test --locked --workspace --lib --no-fail-fast
# Caller supplies the stream and its original ICC; no missing-file skips.
icc-header-repro stream profile:
    cargo run --locked -p zenjxl-decoder --example icc_header_regression -- {{quote(stream)}} {{quote(profile)}}

# Prints CODEC_CORPUS_PATH for corpus-backed feature tests.
feature-corpus:
    cargo run --locked -p zenjxl-decoder --example fetch_feature_corpus

icc-check:
    cargo fmt -p zenjxl-decoder -p zenjxl-decoder-cli -p zenjxl-decoder-macros -p zenjxl-decoder-simd -- --check
    cargo clippy --locked -p zenjxl-decoder --lib --example icc_header_regression --example fetch_feature_corpus -- -D warnings
    cargo test --locked -p zenjxl-decoder --lib icc


# Run through run-heavy on shared hosts; verifies crop/reference u8 parity.
cropped-animation-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- cropped_animation pixel_format_can_change_between_frames --test-threads=1

# Caller supplies a fresh output directory. run-heavy covers build and oracle.
downstream-oracle output:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 4 -- python3 scripts/upstream-audit/verify_downstream.py --output {{quote(output)}}

# Caller wraps with the shared heavy lock and run-heavy on shared hosts.
conformance-limit-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- conformance_limit level5_extra_channel_limit transform_count frame_duration --test-threads=1

# Preserve normal Level 5 CLI coverage while sweeping Level 10 corpus files.
inspector-conformance-check:
    cargo test --locked -p zenjxl-decoder-cli --all-features --bin jxlinspect -- --test-threads=1

# Exact rational and encoded animation-clock boundary regressions.
frame-duration-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib frame_duration -- --test-threads=1

# Retain ordinary seek guarantees while exercising the named reordered file.
downstream-coverage-check:
    cargo clippy --locked --workspace --all-features --all-targets -- -D warnings
    cargo test --locked -p zenjxl-decoder --all-features --lib -- flush_truncated_squeeze_small_tiles squeeze_partial_last_group reordered_jxlp_frames downstream_fixture_u8 seek_every_visible_frame_sweep --test-threads=1

# Caller provides CODEC_CORPUS_PATH and wraps with run-heavy + heavy.lock.
# Constrain CPU affinity at invocation if fixture workers need a lower cap.
downstream-integrated-check:
    python3 scripts/upstream-audit/prepare_feature_references.py --check
    cargo clippy --locked --workspace --all-features --all-targets -- -D warnings
    cargo test --locked -p zenjxl-decoder --all-features --lib threshold_is_inclusive -- --test-threads=1
    cargo fmt -p zenjxl-decoder -p zenjxl-decoder-cli -p zenjxl-decoder-macros -p zenjxl-decoder-simd -- --check
    just downstream-negative-canvas-check
    cargo test --locked --workspace --all-features -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- cropped_animation conformance_limit transform_count incremental_error_tests subsampled_channels_smooth_on_their_own_grid zero_predictor_palette_clips_negative_indices spline_colors_ignore_lf_correlation downstream_fixture_u8 reordered_jxlp_frames squeeze_partial_last_group flush_truncated_squeeze_small_tiles --test-threads=1

# Caller holds heavy.lock and run-heavy; dependencies are cmake/ninja, Brotli,
# LCMS2, PNG and JPEG development packages. Output directory must be fresh.
build-feature-oracle output:
    bash scripts/upstream-audit/build_pinned_reference.sh {{quote(output)}}

# Input cache comes from codec-corpus; write required v0.12 references separately.
feature-corpus-references output:
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 4 -- python3 scripts/upstream-audit/prepare_feature_references.py --output {{quote(output)}}

# Keep debug overflow checks enabled for cropped/padded rows crossing zero.
downstream-negative-canvas-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- wrapped_canvas serial_parallel_parity_palette_across_thread_counts --test-threads=1

# Caller wraps this full-decode diagnostic with heavy.lock and run-heavy.
downstream-tier-check:
    cargo test --locked -p zenjxl-decoder --all-features --test cross_tier_determinism -- decode_is_identical_on_every_dispatch_tier --test-threads=1

# Explicitly enable the CMYK stage when comparing the existing LCMS backend.
downstream-cmyk-dump input output:
    cargo run --locked -p zenjxl-decoder-cli --all-features --features jxl/cms --example cmyk_lcms_reference -- {{quote(input)}} {{quote(output)}}

# Caller wraps with heavy.lock and run-heavy; test the custom CMS without MoxCMS.
downstream-cms-check:
    cargo clippy --locked --workspace --all-features --all-targets -- -D warnings
    cargo test --locked -p zenjxl-decoder-cli --lib -- --test-threads=1
    cargo test --locked -p zenjxl-decoder-cli --no-default-features --lib -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --all-features --lib -- cms_contract moxcms_wrapper custom_cms_receives_documented_reflectance --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- cms_contract custom_cms_receives_documented_reflectance --test-threads=1

# Caller holds heavy.lock and run-heavy. Both WASI feature configurations execute.
downstream-fma-check:
    cargo test --locked -p zenjxl-decoder-simd --all-features --lib fused_multiply_add_contract -- --test-threads=1
    just downstream-tier-check
    cargo test --locked -p zenjxl-decoder-simd --target wasm32-wasip1 --no-default-features --lib fused_multiply_add_contract -- --test-threads=1
    cargo test --locked -p zenjxl-decoder-simd --target wasm32-wasip1 --no-default-features --features wasm128 --lib fused_multiply_add_contract -- --test-threads=1

# Caller holds heavy.lock and run-heavy; optional ZENJXL_DCT_DIAGNOSTICS retains values.
downstream-idct-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib test_idct1d_ -- --test-threads=1

# Caller holds heavy.lock and run-heavy; frozen baseline is from afab87c4.
large-idct-precision-cost group="avx2/idct2d_64_64":
    cargo bench --locked -p zenjxl-decoder --features __bench_kernels --bench large_idct_precision -- --group={{quote(group)}}

# Caller holds heavy.lock and run-heavy. Keep structural and I/O errors strict.
inventory-behavior-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- structural_boxes_after_codestream_remain_invalid trailing_box_io_errors_remain_errors hf_table_budget_tests test_junk_after_the_codestream test_codestream_length_after_a_seek test_scan_needs_no_frame_memory --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- structural_boxes_after_codestream_remain_invalid trailing_box_io_errors_remain_errors hf_table_budget_tests test_junk_after_the_codestream test_codestream_length_after_a_seek test_scan_needs_no_frame_memory --test-threads=1

# Caller holds heavy.lock + run-heavy; binary is the CLI library-test executable.
lcms-context-memcheck binary log:
    valgrind --error-exitcode=99 --leak-check=no --log-file={{quote(log)}} {{quote(binary)}} cms::tests::owned_context_transform_migrates_between_threads --exact --nocapture

# Caller holds heavy.lock and run-heavy; actual migration, not only type assertions.
send-state-check:
    cargo test --locked -p zenjxl-decoder --all-features --test sendable_state
    cargo test --locked -p zenjxl-decoder --no-default-features --test sendable_state
    cargo test --locked -p zenjxl-decoder-cli --all-features --lib decoder_with_lcms_migrates
    cargo test --locked -p zenjxl-decoder-cli --no-default-features --lib decoder_with_lcms_migrates

# Caller holds heavy.lock and run-heavy; metadata access does not alter output.
source-metadata-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- basic_info_profile_guard source_layer_facts frame_duration --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- basic_info_profile_guard source_layer_facts frame_duration --test-threads=1

# Caller supplies a fresh artifact directory and the pinned independent C reader.
source-metadata-oracle artifacts oracle:
    JXL_INTERPRETATION_ARTIFACTS={{quote(artifacts)}} JXL_INTERPRETATION_ORACLE={{quote(oracle)}} cargo test --manifest-path perf/interpretation-validation/Cargo.toml --locked -- --nocapture --test-threads=1

# Caller holds heavy.lock and run-heavy.
operation-contract-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- frame_header_budget stop_reason test_stop_cancellation --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- frame_header_budget stop_reason test_stop_cancellation --test-threads=1

# Caller holds heavy.lock and run-heavy; emitted text is the pinned C++ oracle.
spline-spacing-oracle binary:
    c++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror perf/raw-layer-validation/spline_spacing_oracle.cc -o {{quote(binary)}}
    {{quote(binary)}}

spline-spacing-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- equally_spaced_points noise spline fast_math --test-threads=1

# Caller holds heavy.lock and run-heavy; use a uniform cpuset on hybrid CPUs.
spline-spacing-cost:
    cargo bench --locked --manifest-path perf/spline-precision/Cargo.toml --bench spline_precision

# Full original unit suite; explicit corpus admission, debug checks retained.
stable-cohort-check corpus:
    test -d "{{corpus}}/jxl/features"
    env -u ZENJXL_ALLOW_MISSING_CORPUS CODEC_CORPUS_PATH="{{corpus}}" ZENJXL_TEST_WORKERS=1 RUST_TEST_THREADS=1 CARGO_PROFILE_TEST_OPT_LEVEL=2 CARGO_PROFILE_TEST_DEBUG_ASSERTIONS=true CARGO_PROFILE_TEST_OVERFLOW_CHECKS=true cargo test --locked --workspace --lib -- --nocapture
    cargo clippy --locked --workspace --lib -- -D warnings
