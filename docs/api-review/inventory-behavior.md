# Trailing metadata and scan allocation disposition

PR65 is resolved against current main with its original head retained as a
merge parent. Its compatible fixes have already landed: defer the HF section
table until the TOC, charge all table storage to the memory budget, omit it in
header-only scans, and drain unused finite codestream payloads before reading
valid trailing metadata. Actual compressed-input ICC hints and macOS ARM64 CI
are also retained from main. The older ICC example is superseded by the current
ICC header regression example.

The original blanket trailing-error recovery is excluded. Independent v0.12
accepts some malformed optional metadata but rejects a duplicate structural
ftyp box. The blanket recovery additionally swallows injected I/O failures.
Current regressions retain those errors, along with invalid jxlp-after-jxlc
rejection; accepting every post-codestream error is not a metadata policy.

`just inventory-behavior-check` runs both feature configurations with unchanged
assertions: valid trailing metadata and seek-relative length, no-frame-memory
scans, shared HF table budgeting, structural rejection and I/O propagation.
Those checks, the full workspace and six frozen exact RGBA references passed
in the compatible extraction. The final merge preserves the current runtime
source byte for byte; no previous threshold or assertion is relaxed.
