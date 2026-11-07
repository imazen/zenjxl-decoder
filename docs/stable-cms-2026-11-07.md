# Stable CMS qualification

The optional moxcms requirement moves from 0.9.1 to 0.9.2 in this unmerged
candidate. Numerical decoder kernels and established pixel expectations
remain unchanged. The existing relative-colorimetric CMYK/BPC gate blocks
qualification of the new CMS version.

`just downstream-cmyk-floats` compares the four existing regression inputs
through the existing MoxCMS and LCMS adapters. The negative-red sample
`[0.2347, 0.9601, 0.5918, 0.9813]` produces:

| Source | Red |
| --- | ---: |
| moxcms 0.9.1, original options | -0.06461855 |
| moxcms 0.9.1, fixed-point preference disabled | -0.06435492 |
| moxcms 0.9.2, original options | -0.066765495 |
| Existing CLI LCMS adapter | -0.064411186 |
| Frozen LCMS regression reference | -0.0645 |

The existing test requires absolute error below 0.0013. It passes on 0.9.1
and fails on 0.9.2, with identical source and lock graph except for MoxCMS.
Disabling the old fixed-point preference does not reproduce the new
version's result, so that change alone does not explain the regression.
No threshold, reference value or ignore was changed. All four input rows
and all three output channels are retained in the raw receipts; the older
test intentionally leaves two darker-cyan channels unchecked. These four
samples are a diagnostic, not complete CMYK profile accuracy coverage.

The full existing CMS recipe passes strict all-feature/all-target Clippy
and the CLI library controls, then fails this one of 12 selected decoder
CMS contracts. That run takes 175 seconds, peak RSS 1.59 GiB. The isolated
0.9.1 regression test passes in 20 seconds, peak RSS 1.42 GiB. The float
probes take 10 seconds each, peak RSS 0.78 GiB for 0.9.2 and 0.81 GiB for
0.9.1; the separate old-version floating-grid control takes two seconds,
peak RSS 0.49 GiB. These are validation resource receipts, not speed
comparisons: build work differs between runs.

Default features select SIMD but do not enable the MoxCMS adapter. A full
default-feature workspace library run passes 1,247 tests with 28 existing
ignores, using one test thread and one fixture-sweep worker. Its optimized
test build retains debug assertions and overflow checks, under an 8 GiB
memory cap: 197 seconds, peak RSS 5.45 GiB. Earlier uncapped fixture fanout
inside the capped test scope triggered cgroup OOM even with libtest serial;
those failures are retained and are not successful test runs.

Use the existing `just stable-unit-references CORPUS DJXL FRESH_OUTPUT`
recipe for the four required independent PNG references, then
`just stable-cohort-check OUTPUT`. It reuses the upstream-audit reference
preparer and fixes output at eight bits; no generated images enter git.
The static djxl 0.12 binary identifies revision 4128790, but its linked CMS
backend is not established. The float LCMS comparison above uses the CLI's
explicit LCMS adapter and is independent of those PNG references.

The final run using all four references from the existing eight-bit preparer
also passes all 1,247 default-feature workspace library tests, with the same
28 ignores, strict all-feature/all-target workspace Clippy and formatting.
It takes 199 seconds, peak RSS 5.48 GiB, minimum available 13333 MiB and
peak load 1.29. The optional CMYK float failure above remains a blocker.
