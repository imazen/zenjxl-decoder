// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Mirror-reflects a value v to fit in a [0; s) range.
pub fn mirror(mut v: isize, s: usize) -> usize {
    // An empty range has nothing to mirror into; the loop below would never
    // terminate (v flips between -1 and 0 forever). Callers must not pass
    // s == 0, so fail loudly in debug builds and stay finite in release.
    debug_assert!(s > 0, "mirror() into an empty range");
    if s == 0 {
        return 0;
    }
    // TODO(veluca): consider speeding this up if needed.
    loop {
        if v < 0 {
            v = -v - 1;
        } else if v >= s as isize {
            v = s as isize * 2 - v - 1;
        } else {
            return v as usize;
        }
    }
}
