# Spline point-spacing cost

This standalone test tool compares complete one-shot decodes against main
`6d060afb26462780c3690c771d7d3674dbd23ebf`. Both dependencies disable default
features, use the same default decoder options, and decode the existing basic
and layered noise/spline fixtures. Input creation and returned-image destruction
follow zenbench's shared timing boundary. This is a two-fixture cost check,
not a general decoder performance claim.

Run `just spline-spacing-cost` under the shared heavy lock and resource wrapper.
Hybrid CPUs additionally need a uniform cgroup cpuset; process affinity alone
does not satisfy zenbench's comparison policy. No native CPU compilation flags
are used. The lockfile pins the benchmark dependencies.

The independent point oracle is
[`../raw-layer-validation/spline_spacing_oracle.cc`](../raw-layer-validation/spline_spacing_oracle.cc).
It preserves the scalar point-walking operation order from
[libjxl v0.12 splines.cc](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/splines.cc).
`just spline-spacing-oracle <binary>` emits the exact float-bit constants used
by `equally_spaced_points_match_pinned_reference_bits`. Compile it without
floating-point contraction, as the recipe specifies. `just spline-spacing-check`
runs that regression and the existing noise, spline and fast-math checks.

The repair changes the operation order used to locate spline samples. It does
not change the high-precision option, its default, the spline cutoff, rendering
approximations, or any test tolerance. Exact point placement alone does not
establish bit-exact raw RGB layers; that qualification remains on draft PR75.
