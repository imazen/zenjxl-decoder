#!/bin/sh
# Runs a wasm32-wasip1 test binary under wasmtime. Cargo runs tests with the
# package directory as the working directory; the tests locate fixtures with
# absolute paths built from `env!("CARGO_MANIFEST_DIR")`, so the host package
# directory is mapped into the guest at the same absolute path (plus its
# parent, for the workspace-level `tests/testdata` fixtures).
#
# wasmtime starts the guest with an EMPTY environment: every variable the tests
# read has to be named here. `ZENJXL_ALLOW_MISSING_CORPUS` is the explicit
# "corpus is not provisioned" flag CI sets at the workflow level — without it
# the corpus-backed feature tests fail loudly by design (no silent skips), which
# is what turned the wasm legs red from 2026-08-27 on.
set -eu
here=$(pwd)
exec wasmtime --dir="$here::$here" --dir="$(dirname "$here")::$(dirname "$here")" \
  --dir=.::. --env ARBTEST_BUDGET_MS --env ZENJXL_ALLOW_MISSING_CORPUS "$@"
