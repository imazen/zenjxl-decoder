#!/bin/sh
# Runs a wasm32-wasip1 test binary under wasmtime. Cargo runs tests with the
# package directory as the working directory; the tests locate fixtures with
# absolute paths built from `env!("CARGO_MANIFEST_DIR")`, so the host package
# directory is mapped into the guest at the same absolute path (plus its
# parent, for the workspace-level `tests/testdata` fixtures).
set -eu
here=$(pwd)
exec wasmtime --dir="$here::$here" --dir="$(dirname "$here")::$(dirname "$here")" \
  --dir=.::. --env ARBTEST_BUDGET_MS "$@"
