// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::{BufRead, BufReader, Error, IoSliceMut, Read, Seek, SeekFrom};

pub trait JxlBitstreamInput {
    /// Returns an estimate bound of the total number of bytes that can be read via `read`.
    /// Returning a too-low estimate here can impede parallelism. Returning a too-high
    /// estimate can increase memory usage.
    fn available_bytes(&mut self) -> Result<usize, Error>;

    /// Fills in `bufs` with more bytes, returning the number of bytes written.
    /// Buffers are filled in order and to completion.
    fn read(&mut self, bufs: &mut [IoSliceMut]) -> Result<usize, Error>;

    /// Skips up to `bytes` bytes of input. The provided implementation just uses `read`, but in
    /// some cases this can be implemented faster.
    /// Returns the number of bytes that were skipped. If this returns 0, it is assumed that no
    /// more input is available.
    fn skip(&mut self, bytes: usize) -> Result<usize, Error> {
        let mut bytes = bytes;
        const BUF_SIZE: usize = 1024;
        let mut skip_buf = [0; BUF_SIZE];
        let mut skipped = 0;
        while bytes > 0 {
            let num = bytes.min(BUF_SIZE);
            // Count what `read` actually returned: a short read (normal for
            // sockets and pipes) used to be counted as a full `num`, so the
            // skip reported more than it consumed and the stream desynced.
            let n = self.read(&mut [IoSliceMut::new(&mut skip_buf[..num])])?;
            if n == 0 {
                break;
            }
            bytes -= n;
            skipped += n;
        }
        Ok(skipped)
    }
}

impl JxlBitstreamInput for &[u8] {
    fn available_bytes(&mut self) -> Result<usize, Error> {
        Ok(self.len())
    }

    fn read(&mut self, bufs: &mut [IoSliceMut]) -> Result<usize, Error> {
        self.read_vectored(bufs)
    }

    fn skip(&mut self, bytes: usize) -> Result<usize, Error> {
        let num = bytes.min(self.len());
        self.consume(num);
        Ok(num)
    }
}

impl<R: Read + Seek> JxlBitstreamInput for BufReader<R> {
    fn available_bytes(&mut self) -> Result<usize, Error> {
        let pos = self.stream_position()?;
        let end = self.seek(SeekFrom::End(0))?;
        self.seek(SeekFrom::Start(pos))?;
        Ok(end.saturating_sub(pos) as usize)
    }

    fn read(&mut self, bufs: &mut [IoSliceMut]) -> Result<usize, Error> {
        self.read_vectored(bufs)
    }

    fn skip(&mut self, bytes: usize) -> Result<usize, Error> {
        let cur = self.stream_position()?;
        // `bytes as i64` turned very large skips negative -- `usize::MAX`
        // became a one-byte seek *backwards*.
        if let Ok(offset) = i64::try_from(bytes) {
            self.seek(SeekFrom::Current(offset))
                .map(|x| x.saturating_sub(cur) as usize)
        } else {
            // Beyond i64::MAX: skipping that far means "to the end"; a clamped
            // Current(i64::MAX) seek fails with EINVAL on file descriptors.
            // Upstream jxl-rs c1e2e3d.
            self.seek(SeekFrom::End(0))
                .map(|x| x.saturating_sub(cur) as usize)
        }
    }
}

#[cfg(test)]
mod skip_tests {
    use super::JxlBitstreamInput;
    use std::io::{BufReader, Cursor, Error, IoSliceMut, Read, Seek};

    /// Returns at most 3 bytes per read, like a socket delivering small
    /// packets.
    struct Trickle<'a>(&'a [u8]);
    impl JxlBitstreamInput for Trickle<'_> {
        fn available_bytes(&mut self) -> Result<usize, Error> {
            Ok(self.0.len())
        }
        fn read(&mut self, bufs: &mut [IoSliceMut]) -> Result<usize, Error> {
            let n = bufs[0].len().min(3).min(self.0.len());
            bufs[0][..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }

    /// The default `skip` must report and consume exactly what the reader
    /// delivered. It used to count every short read as a full chunk. Upstream
    /// jxl-rs f809996.
    #[test]
    fn default_skip_counts_short_reads() {
        let data: Vec<u8> = (0..100).collect();
        let mut t = Trickle(&data);
        assert_eq!(t.skip(10).unwrap(), 10);
        let mut next = [0u8; 1];
        t.read(&mut [IoSliceMut::new(&mut next)]).unwrap();
        assert_eq!(next[0], 10, "the byte after a 10-byte skip");

        let short = [1u8, 2, 3, 4, 5];
        assert_eq!(Trickle(&short).skip(1000).unwrap(), 5, "only 5 bytes exist");
    }

    /// `usize::MAX` must seek forwards (to EOF), not one byte backwards.
    #[test]
    fn bufreader_skip_huge_does_not_seek_backwards() {
        let mut r = BufReader::new(Cursor::new(vec![0u8; 64]));
        let mut b = [0u8; 16];
        r.read_exact(&mut b).unwrap();
        let before = r.stream_position().unwrap();
        let _ = r.skip(usize::MAX);
        assert!(
            r.stream_position().unwrap() >= before,
            "skip must never move backwards"
        );
    }
}
