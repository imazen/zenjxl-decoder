#!/usr/bin/env python3
"""Provide two missing independent references without editing codec-corpus.

Usage: prepare-unit-reference-controls.py CORPUS DJXL NEW_OUTPUT
All original corpus input categories remain visible through read-only symlinks.
Only the two references required by ordinary diagnostic tests are generated;
this does not qualify the separately ignored full corpus parity suite.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    corpus, decoder, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
    assert (corpus / "jxl/features").is_dir()
    assert decoder.is_file()
    output.mkdir(parents=True, exist_ok=False)
    root = output / "jxl"
    root.mkdir()
    categories = ["conformance", "features", "photographic", "edge-cases"]
    for category in categories:
        source = corpus / "jxl" / category
        assert source.is_dir(), source
        (root / category).symlink_to(source, target_is_directory=True)
    identity = {
        "oracle_sha256": digest(decoder),
        "oracle_version": subprocess.check_output([str(decoder), "--version"],
                                                   text=True, stderr=subprocess.STDOUT),
        "corpus_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=corpus, text=True).strip(),
        "scope": "two independent references; all inputs retained; no existing golden replaced",
        "references": [],
    }
    for name in ["cmyk_layers", "multiple_layers_noise_spline"]:
        matches = [path for category in categories
                   for path in (corpus / "jxl" / category).glob(name + ".jxl")]
        assert len(matches) == 1, matches
        source = matches[0]
        target = root / "reference" / source.parent.name / (name + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [str(decoder), str(source), str(target), "--num_threads=2"]
        process = subprocess.run(command, text=True, capture_output=True, check=True)
        (output / (name + ".log")).write_text(process.stdout + process.stderr)
        identity["references"].append({
            "input": str(source.relative_to(corpus)), "input_sha256": digest(source),
            "output": str(target.relative_to(output)), "output_sha256": digest(target),
            "output_bytes": target.stat().st_size,
        })
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
