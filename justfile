# zenjxl-decoder dev commands

# Caller holds the shared heavy lock and run-heavy; optional target exercises i686.
upstream-regression-check target="" filter="upstream_2026_11" profile="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=()
    if [[ -n '{{target}}' ]]; then args+=(--target '{{target}}'); fi
    if [[ '{{profile}}' == release ]]; then args+=(--release); fi
    cargo test --locked -p zenjxl-decoder --all-features --lib "${args[@]}" '{{filter}}' -- --test-threads=1

# Caller holds the heavy-job lock and resource wrapper.
upstream-foundation-check:
    just upstream-regression-check '' bit_reader
    just upstream-regression-check '' util::float16
    just upstream-regression-check '' test_frame_dimension_checks

# Caller supplies CODEC_CORPUS_PATH, heavy.lock and the resource wrapper.
upstream-import-workspace-check:
    python3 scripts/upstream-audit/prepare_feature_references.py --check
    cargo test --locked --release --workspace --all-features -- --test-threads=1

upstream-import-minimal-check:
    cargo clippy --locked -p zenjxl-decoder --no-default-features --lib -- -D warnings
    cargo test --locked --release -p zenjxl-decoder --no-default-features --lib -- upstream_ container::gain_map review_color_ review_dci test_out_of_bounds_reports_bytes test_apply_patch_delta_bounds test_patch_area_limit rect_byte_width_overflow rect_large_coord_clip hlg_ootf test_frame_dimension_checks alphabet_allocation_guard --test-threads=1

# Caller supplies CODEC_CORPUS_PATH, heavy.lock and run-heavy (14G / four jobs).
review-final-check output reference djxl revision="":
    just upstream-import-workspace-check
    just upstream-import-minimal-check
    cargo clippy --locked --workspace --all-features --all-targets -- -D warnings
    cargo test --locked -p zenjxl-decoder-cli --no-default-features --lib -- --test-threads=1
    cargo run --locked -p zenjxl-decoder-cli --no-default-features --example hdr_intensity_probe
    just upstream-frozen-oracle {{quote(output)}} {{quote(reference)}} {{quote(djxl)}} {{quote(revision)}}

# Exact upstream coordinate regressions; use debug for checked-arithmetic control.
upstream-boundary-check profile="release":
    #!/usr/bin/env bash
    set -euo pipefail
    args=()
    if [[ '{{profile}}' == release ]]; then args+=(--release); fi
    cargo test --locked -p zenjxl-decoder --all-features --lib "${args[@]}" -- rect_byte_width_overflow rect_large_coord_clip test_apply_patch_delta_bounds --test-threads=1

upstream-followup-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- alphabet_allocation_guard frame_header_budget_counts_low_frequency_frames tests::hdr:: hlg_ootf_arb fuzzer_context_map_num_histograms_overflow modular_rle_fast_path_matches_without_lz77 --test-threads=1

# Read a completed job while the rest of its workflow is still running.
review-validation-check filter="conformance_limit_tests":
    cargo test --locked -p zenjxl-decoder --all-features --lib '{{filter}}' -- --test-threads=1

review-level-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- conformance_limit_tests icc_output_limit --test-threads=1

review-prefix-check:
    cargo test --locked --release -p zenjxl-decoder --all-features --lib -- partial_decode jxlrs_testdata_ports --test-threads=1

review-reference-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- before_ct_extra stages::blending --test-threads=1

review-state-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- cumulative_zero_duration upstream_seek_matches_sequential test_start_new_frame frame_header_budget --test-threads=1

review-metadata-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- gain_map_box_cardinality gain_map_replay required_external_jpeg black_channel_requires wire_black --test-threads=1

# Diagnostic defaults to measurements; --check retains upstream's HDR assertions.
review-lcms-intensity *args:
    cargo run --locked -p zenjxl-decoder-cli --example hdr_intensity_probe -- {{args}}

# Approved public profile method, external HDR control and exact CMYK goldens.
review-intensity-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib intensity_profile_conversion -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib intensity_profile_conversion -- --test-threads=1
    cargo test --locked -p zenjxl-decoder-cli --all-features --lib -- --test-threads=1
    cargo test --locked -p zenjxl-decoder-cli --no-default-features --lib -- --test-threads=1
    cargo run --locked -p zenjxl-decoder-cli --example hdr_intensity_probe -- --check
    cargo run --locked -p zenjxl-decoder-cli --no-default-features --example hdr_intensity_probe -- --check
    cargo clippy --locked --workspace --all-features --all-targets -- -D warnings

ci-job-log job output:
    gh api 'repos/{owner}/{repo}/actions/jobs/{{job}}/logs' --allow-escape-sequences > {{quote(output)}}

# Run with the host's heavy-job wrapper.
review-ma-tree-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib ma_tree_ -- --test-threads=1

# Use an existing v0.12.0 source/build pair. Outputs are caller-owned evidence.
review-native-build source build binary:
    {{quote(build)}}/tools/cjxl --version | rg 'v0\.12\.0 a7a9c78'
    TMPDIR="$HOME/tmp" c++ -std=c++17 -O2 -I{{quote(source)}}/lib/include -I{{quote(build)}}/lib/include scripts/remaining_native.cc -L{{quote(build)}}/lib -Wl,-rpath,{{quote(build)}}/lib -ljxl -o {{quote(binary)}}

review-native-oracle binary output:
    python3 scripts/remaining_native_cases.py {{quote(binary)}} {{quote(output)}}

review-native-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- native_remaining hf_coefficient_pass_and_sum_limits wire_black_channel_requires_level10 --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- native_remaining hf_coefficient_pass_and_sum_limits wire_black_channel_requires_level10 --test-threads=1

review-prefix-alphabet-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib entropy_coding::huffman -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib entropy_coding::huffman -- --test-threads=1

review-noise-check:
    cargo test --locked -p zenjxl-decoder --all-features --lib -- native_remaining_sprite_noise noise_process_row_chunk --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- native_remaining_sprite_noise noise_process_row_chunk --test-threads=1

review-seek-check:
    cargo test --locked --release -p zenjxl-decoder --all-features --lib seek -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib seek -- --test-threads=1

upstream-test-inventory upstream:
    python3 scripts/upstream-audit/inventory_tests.py {{quote(upstream)}}

# Caller supplies the heavy-job lock/resource wrapper and a fresh output path.
upstream-frozen-oracle output reference djxl revision="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=()
    if [[ -n {{quote(revision)}} ]]; then args+=(--decoder-revision {{quote(revision)}}); fi
    python3 scripts/upstream-audit/verify_downstream.py --output {{quote(output)}} --reference-directory {{quote(reference)}} --djxl {{quote(djxl)}} "${args[@]}"

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

# Paired comparisons of two pinned CLI release binaries with identical output.
paired-u8 output baseline candidate *args:
    python3 scripts/perf/paired_u8.py {{quote(output)}} {{quote(baseline)}} {{quote(candidate)}} {{args}}

# Compare the still-image API with caller-owned output buffers and rewind.
build-paired-output:
    cargo build --locked --release -p zenjxl-decoder-cli --bin paired-output --no-default-features --features all-simd,threads

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

# Retained before/after benchmark binaries; invoke through run-heavy on an idle host.
allocation-decode-measure binary label threads group="":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$HOME/tmp/decoder-issue18/logs"
    run_label={{quote(label)}}
    thread_count={{quote(threads)}}
    group_filter={{quote(group)}}
    args=()
    if [[ -n "$group_filter" ]]; then args+=("--group=$group_filter"); fi
    RAYON_NUM_THREADS="$thread_count" {{quote(binary)}} "${args[@]}" --format=json > "$HOME/tmp/decoder-issue18/logs/${run_label}-t${thread_count}.json" 2> "$HOME/tmp/decoder-issue18/logs/${run_label}-t${thread_count}.log"

allocation-check:
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
    cargo test --locked --release -p zenjxl-decoder --all-features --lib image::
    cargo test --locked --release -p zenjxl-decoder --no-default-features --lib image::
    cargo test --locked --release -p zenjxl-decoder --all-features --test cross_tier_determinism --test modular_buffer_recycling --test output_buffer_limit

allocation-status:
    #!/usr/bin/env bash
    set -euo pipefail
    for log in "$HOME/tmp/decoder-issue18/logs/"*.log; do
        echo "$log"
        tail -n 4 "$log"
    done
    df -h "$HOME/tmp"

allocation-summary logs:
    python3 scripts/perf/allocation_summary.py {{quote(logs)}}

# Exact embedded issue-144 source/stream; no external corpus is needed.
global-squeeze-regression:
    cargo clippy --locked -p zenjxl-decoder --lib -- -D warnings
    cargo test --locked -p zenjxl-decoder --test global_squeeze_grid -- --nocapture --test-threads=1

global-squeeze-mutation revision output:
    python3 scripts/check_global_squeeze_mutation.py {{quote(revision)}} {{quote(output)}}

# Frozen feature inputs and the two pinned-reference PNGs must be supplied.
global-squeeze-integrated:
    python3 scripts/upstream-audit/prepare_feature_references.py --check
    cargo fmt -p zenjxl-decoder -p zenjxl-decoder-cli -p zenjxl-decoder-macros -p zenjxl-decoder-simd -- --check
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
    cargo test --locked --release --workspace --all-features -- --test-threads=1
    cargo test --locked --release --workspace --no-default-features -- --test-threads=1
    cargo test --locked --release --workspace --no-default-features --features threads,all-simd -- --test-threads=1

# Retain the target and profiles for selected make_grids region inspection.
global-squeeze-coverage:
    cargo llvm-cov --locked --no-report -p zenjxl-decoder --features threads --test global_squeeze_grid -- --test-threads=1

global-squeeze-regions llvm_directory executable profile output:
    python3 scripts/export_global_squeeze_coverage.py {{quote(llvm_directory)}} {{quote(executable)}} {{quote(profile)}} {{quote(output)}}

# Match the compiler that diagnosed the post-squeeze CI lint failures.
ci-iterator-check:
    cargo +1.100.0 fmt -p zenjxl-decoder -p zenjxl-decoder-cli -p zenjxl-decoder-macros -p zenjxl-decoder-simd
    cargo +1.100.0 clippy --locked --release --workspace --all-targets --all-features -- -D warnings
    cargo +1.100.0 clippy --locked --release --workspace --all-targets --no-default-features -- -D warnings
    cargo +1.100.0 test --locked --release --workspace --all-features -- --test-threads=1

ci-iterator-toolchain:
    rustup component add --toolchain 1.100.0 rustfmt clippy
    just ci-iterator-check

# The PR source/license checks, without compiling.
source-check:
    ./ci/pull_request_checks.sh

# Native v0.12 empty-DHT / zero-symbol distinction; caller uses run-heavy.
jpeg-dht-contract-check:
    cargo test --locked --release -p zenjxl-decoder --features jpeg --test jpeg_dht_contract -- --nocapture --test-threads=1

jpeg-dht-reader-check:
    cargo test --locked --release -p zenjxl-decoder --features jpeg --lib native_dht_ -- --nocapture --test-threads=1

jpeg-dht-candidate-check:
    cargo fmt -p zenjxl-decoder
    just jpeg-dht-reader-check
    just jpeg-dht-contract-check

jpeg-dht-retained-evidence manifest output:
    python3 scripts/jpeg_dht_evidence.py {{quote(manifest)}} {{quote(output)}}

jpeg-dht-integrated output:
    cargo fmt -p zenjxl-decoder
    just jpeg-dht-retained-evidence Cargo.toml {{quote(output)}}
    just global-squeeze-integrated
    just source-check

# Checked and unchecked reconstruction boundaries, plus explicit scalar JPEG.
jpeg-dht-focused:
    cargo fmt -p zenjxl-decoder
    just jpeg-dht-reader-check
    cargo test --locked --release -p zenjxl-decoder --features jpeg --lib malformed_native_jbrd_error_survives_unchecked_take -- --nocapture --test-threads=1
    just jpeg-dht-contract-check
    cargo test --locked --release -p zenjxl-decoder --no-default-features --features jpeg --test jpeg_dht_contract -- --nocapture --test-threads=1
    cargo clippy --locked -p zenjxl-decoder --all-targets --all-features -- -D warnings
    cargo clippy --locked -p zenjxl-decoder --all-targets --no-default-features --features jpeg -- -D warnings
    just source-check

# Use a distinct instrumented Cargo target; keep the JSON, objects and profiles.
jpeg-dht-coverage output:
    python3 scripts/jpeg_dht_coverage.py {{quote(output)}}

jpeg-dht-mutations output:
    python3 scripts/jpeg_dht_mutations.py {{quote(output)}}

jpeg-scan-contract-check:
    cargo test --locked -p zenjxl-decoder --features jpeg --test jpeg_scan_contract -- --test-threads=1

jpeg-scan-pixel-controls native_run output:
    python3 scripts/jpeg_scan_pixel_controls.py {{quote(native_run)}} {{quote(output)}}

jpeg-scan-native-controls cjxl djxl output:
    python3 scripts/jpeg_scan_native_controls.py {{quote(cjxl)}} {{quote(djxl)}} {{quote(output)}}

jpeg-scan-focused:
    cargo test --locked -p zenjxl-decoder --features jpeg --test jpeg_scan_contract --test jpeg_dht_contract -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --features jpeg --lib baseline_trailing_zrl_ -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --features jpeg --test jpeg_scan_contract -- --test-threads=1
    cargo clippy --locked -p zenjxl-decoder --all-targets --all-features -- -D warnings
    cargo fmt -p zenjxl-decoder --check

jpeg-scan-coverage output:
    python3 scripts/jpeg_scan_coverage.py {{quote(output)}}

jpeg-scan-mutations output:
    python3 scripts/jpeg_scan_mutations.py {{quote(output)}}

jpeg-scan-regions output:
    python3 scripts/jpeg_scan_coverage.py {{quote(output)}} --report-only

jpeg-scan-integrated:
    just source-check
    just global-squeeze-integrated
    just jpeg-scan-focused

jpeg-color-native-controls cjxl djxl output:
    python3 scripts/jpeg_color_native_controls.py {{quote(cjxl)}} {{quote(djxl)}} {{quote(output)}}

jpeg-color-retained-evidence manifest output:
    python3 scripts/jpeg_color_evidence.py {{quote(manifest)}} {{quote(output)}}

jpeg-color-integrated:
    just source-check
    python3 scripts/upstream-audit/prepare_feature_references.py --check
    cargo test --locked --workspace --all-features -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --features jpeg --test jpeg_scan_contract --test jpeg_dht_contract -- --test-threads=1
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
    cargo fmt -p zenjxl-decoder -p zenjxl-decoder-cli -p zenjxl-decoder-macros -p zenjxl-decoder-simd --check

jpeg-color-original-replay archive output:
    python3 scripts/jpeg_color_original_replay.py {{quote(archive)}} {{quote(output)}}

jpeg-color-fixture-text native:
    python3 scripts/jpeg_color_fixture_text.py {{quote(native)}}

# Caller supplies run-heavy and immutable v0.12 source/build roots.
hlg-ootf-native source build output binary:
    python3 scripts/hlg_ootf_oracle.py --binary {{quote(binary)}} {{quote(source)}} {{quote(build)}} {{quote(output)}}

hlg-near-unity-check:
    cargo test --locked -p zenjxl-decoder --lib hlg_near_unity_ -- --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib hlg_near_unity_ -- --test-threads=1

# Target color/stage callers without repeating unrelated extended prefix campaigns.
hlg-near-unity-broad-check:
    cargo test --locked -p zenjxl-decoder --lib -- color:: render::stages::from_linear render::stages::to_linear tests::hdr:: --test-threads=1
    cargo test --locked -p zenjxl-decoder --no-default-features --lib -- color:: render::stages::from_linear render::stages::to_linear tests::hdr:: --test-threads=1
    cargo clippy --locked -p zenjxl-decoder --all-features --lib --tests -- -D warnings
    cargo fmt -p zenjxl-decoder --check

hlg-forward-native source build output binary:
    python3 scripts/hlg_forward_fixture.py {{quote(source)}} {{quote(build)}} {{quote(output)}} {{quote(binary)}}

hlg-cutoff-fault output:
    python3 scripts/hlg_ootf_mutation.py {{quote(output)}}

profile-admission-fault output:
    python3 scripts/profile_admission_mutation.py {{quote(output)}}

decoder-regions llvm archive output kind:
    python3 scripts/decoder_regions.py {{quote(llvm)}} {{quote(archive)}} {{quote(output)}} --kind {{quote(kind)}}
