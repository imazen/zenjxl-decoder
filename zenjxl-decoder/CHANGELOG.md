# Changelog

All notable changes to `zenjxl-decoder` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this crate uses 0.x semver
(minor = breaking, patch = additive/fixes).

## [Unreleased]

### Added
- A full crate README: quick start, the server-safety story (resource-limit
  presets + cooperative cancellation), all entry points, the feature matrix, and
  the `#[non_exhaustive]` builder idiom for `JxlDecoderOptions`. The
  conventions-compliant repo-root `README.md` is the source; the crates.io page
  renders a generated, CI-badge-only `README.crates.md`
  (`readme = "../README.crates.md"`).

### Changed
- `JxlDecoderLimits::restrictive()` raises `max_pixels` from 100 MP to **120 MP**
  so common ~108 MP camera photos pass an untrusted-web policy while >120 MP
  decompression bombs are still rejected.
