# Movable decoder states

The proposed public change is `pub trait JxlCms: Send`. It requires movable
providers, while transformer objects keep their existing `Send + Sync` bound.
Providers do not need concurrent access. Non-Send provider implementations
stop compiling; this breaking bound remains an owner review decision.

LCMS is already a Send provider. Its real transforms now retain their actual
ThreadContext in a private mutex, with transform destruction first (main
`552da568`). Context ownership alone cannot make `Box<dyn JxlCms>` Send: the
trait object must promise it.

The original three-frame migration fixture and sample assertions are retained
unchanged. An additional CLI regression carries the real LCMS-backed decoder
through image/frame/render states on separate threads, then drops it on another
thread. Its entire 512×512 RGBA image retains the existing djxl CMYK hash
`0xd931c4898c628de9`.

Both tests pass with all features and no default features. Strict all-target
all-feature workspace Clippy passes. The independent context-ownership test
also passes Memcheck. The baseline identity test has zero Memcheck errors;
no invalid-memory-access reproduction is claimed.

This refresh uses main `552da568` and preserves the original PR head as a merge
parent. It drops the old coverage-workflow changes; current main's explicit
corpus policy remains in effect. No precision kernel or pixel expectation is
changed. The 735-byte animation fixture is the existing PR60 blob unchanged.
