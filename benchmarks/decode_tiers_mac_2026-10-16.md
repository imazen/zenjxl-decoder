# JPEG XL decode tiers on macOS ARM64, 2026-10-16

Source `231d7710`, M4 Pro. Command: `~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-decode-tiers-macos`. The recipe ran `cargo bench -p zenjxl-decoder --bench decode_bench -- --group=decode_tiers --cpu-policy=unrestricted --format=llm`. Full output is `~/tmp/zenjxl-decode-tiers.log` on this Mac. The benchmark interleaves complete decodes with NEON enabled and forced scalar; the scheduler was unrestricted.

| Fixture | NEON mean | Forced scalar mean | Scalar versus NEON |
| --- | ---: | ---: | ---: |
| `cafe_web_q80.jxl` | 32.14 ms | 43.57 ms | +35.57%, CI +32.33% to +38.93% |
| `portrait_4k_q75.jxl` | 363.27 ms | 846.13 ms | +132.92%, CI +132.32% to +133.50%; nine rounds |
| `green_queen_modular_e3.jxl` | 17.44 ms | 17.45 ms | +1.34%, CI −3.13% to +5.84%; nonsignificant |
| `gray_alpha_lossless.jxl` | 383.96 µs | 390.19 µs | +1.23%, CI −2.05% to +4.48%; nonsignificant |

NEON continues to make a large difference for the two VarDCT fixtures. The two modular comparisons remain ties within measurement uncertainty, consistent with the earlier [ARM audit](arm_audit_2026-09-06/README.md). The absolute times differ from that audit; its older source and toolchain prevent attributing the difference to any one change. A modular call-stack profile and a rejected array experiment are documented there, so this measurement alone supports no new modular optimization.

`run-heavy`: peak RSS 1.43 GiB, minimum available memory 5304 MiB, peak load 2.81.
