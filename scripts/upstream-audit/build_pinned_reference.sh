#!/usr/bin/env bash
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
set -euo pipefail

# Test oracle only. The caller supplies a fresh, disk-backed output directory.
reference_work=$(realpath -m -- "${1:?usage: build_pinned_reference.sh fresh-output-directory}")
mkdir -- "$reference_work"
mkdir -- "$reference_work/scratch"
export TMPDIR="$reference_work/scratch"
reference_revision=a7a9c787341cf703dede03c2009fa460cae5e5df
highway_revision=457c891775a7397bdb0376bb1031e6e027af1c48
fetch_archive() {
    local name=$1 url=$2 checksum=$3
    if [[ -n ${JXL_ORACLE_ARCHIVE_CACHE:-} ]]; then
        cp -- "$JXL_ORACLE_ARCHIVE_CACHE/$name.tar.gz" "$reference_work/$name.tar.gz"
    else
        curl --fail --location --retry 2 "$url" -o "$reference_work/$name.tar.gz"
    fi
    printf '%s  %s\n' "$checksum" "$reference_work/$name.tar.gz" | sha256sum --check --status
}
fetch_archive libjxl "https://codeload.github.com/libjxl/libjxl/tar.gz/$reference_revision" 818398895831069902e3677d285054a7d1255b11b221e94c6aaa1cb83b0a3f29
fetch_archive highway "https://codeload.github.com/google/highway/tar.gz/$highway_revision" 5124b0501c98d9930dbb065bfa1a5bbbd59ce0f12facb7e1e33aaef01a5f1f1a
mkdir -- "$reference_work/source"
tar -xzf "$reference_work/libjxl.tar.gz" --strip-components=1 -C "$reference_work/source"
mkdir -p -- "$reference_work/source/third_party/highway"
tar -xzf "$reference_work/highway.tar.gz" --strip-components=1 -C "$reference_work/source/third_party/highway"
cmake -S "$reference_work/source" -B "$reference_work/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_TOOLS=ON -DJPEGXL_ENABLE_DEVTOOLS=OFF \
    -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF \
    -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_SJPEG=OFF \
    -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_SKCMS=OFF \
    -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_MANPAGES=OFF \
    -DJPEGXL_ENABLE_VIEWERS=OFF -DJPEGXL_ENABLE_PLUGINS=OFF \
    -DJPEGXL_ENABLE_TCMALLOC=OFF -DJPEGXL_ENABLE_LTO=OFF \
    -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
    -DHWY_ENABLE_TESTS=OFF -DHWY_ENABLE_EXAMPLES=OFF
cmake --build "$reference_work/build" --target djxl --parallel "${JXL_ORACLE_BUILD_JOBS:-2}"
"$reference_work/build/tools/djxl" --version > "$reference_work/version.txt" 2>&1
grep -F 'v0.12.0 ' "$reference_work/version.txt"
{
    printf 'libjxl_commit=%s\nhighway_commit=%s\n' "$reference_revision" "$highway_revision"
    sha256sum "$reference_work/libjxl.tar.gz" "$reference_work/highway.tar.gz" "$reference_work/build/tools/djxl"
    pkg-config --modversion lcms2 libbrotlidec libpng
} > "$reference_work/provenance.txt"
