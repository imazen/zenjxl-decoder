// Probe: does "feed data incrementally to process() and interrupt between calls"
// actually bound cancellation latency on jxl-rs main?
use std::io::IoSliceMut;
use std::time::{Duration, Instant};

use jxl::api::{
    Endianness, JxlBitstreamInput, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlParallelRunner, JxlParallelRunnerFun, ProcessingResult,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

static RUN_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TASK_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct RayonRunner;
impl JxlParallelRunner for RayonRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun) -> jxl::error::Result<()> {
        RUN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (0..num).into_par_iter().try_for_each(|i| {
            TASK_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            fun(i)
        })
    }
    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

/// Input that hands out at most `cap` bytes per process() call.
struct Capped<'a> {
    data: &'a [u8],
    pos: usize,
    budget: usize,
    honest_avail: bool,
}
impl<'a> JxlBitstreamInput for Capped<'a> {
    fn available_bytes(&mut self) -> Result<usize, std::io::Error> {
        if self.honest_avail {
            Ok(self.data.len() - self.pos)
        } else {
            Ok(self.budget.min(self.data.len() - self.pos))
        }
    }
    fn read(&mut self, bufs: &mut [IoSliceMut]) -> Result<usize, std::io::Error> {
        let mut total = 0;
        for b in bufs.iter_mut() {
            let n = b.len().min(self.budget).min(self.data.len() - self.pos);
            if n == 0 {
                break;
            }
            b[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            self.budget -= n;
            total += n;
        }
        Ok(total)
    }
}

fn run(path: &str, chunk: Option<usize>, threads: bool, trace: bool, honest_avail: bool) -> (Duration, Duration, usize, (usize, usize)) {
    let mut timeline: Vec<(usize, usize, f64, &'static str)> = vec![];
    let data = std::fs::read(path).unwrap();
    let opts = JxlDecoderOptions::default();
    let mut dec = JxlDecoder::<jxl::api::states::Initialized>::new(opts);
    let cap = chunk.unwrap_or(usize::MAX);
    let mut pos = 0usize;
    let mut calls = 0usize;
    let mut max_call = Duration::ZERO;
    let t0 = Instant::now();

    macro_rules! mkinput {
        () => {
            Capped { data: &data, pos, budget: cap, honest_avail }
        };
    }
    RUN_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    TASK_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut rr = RayonRunner;
    macro_rules! runner {
        () => {
            if threads { Some(&mut rr as &mut dyn JxlParallelRunner) } else { None }
        };
    }

    // stage 1: image info
    let mut d1 = loop {
        let mut inp = mkinput!();
        let t = Instant::now();
        let r = dec.process(&mut inp, runner!()).unwrap();
        let e = t.elapsed();
        max_call = max_call.max(e);
        calls += 1;
        pos = inp.pos;
        timeline.push((calls, pos, e.as_secs_f64()*1e3, "info"));
        match r {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(pos < data.len(), "truncated");
                dec = fallback;
            }
        }
    };

    let size = (d1.basic_info().size.0 as usize, d1.basic_info().size.1 as usize);
    let necs = d1.current_pixel_format().extra_channel_format.len();
    let pf = jxl::api::JxlPixelFormat::rgba8(necs);
    let nchan = 4usize;
    d1.set_pixel_format(pf).unwrap();
    let mut buf = vec![0u8; size.0 * size.1 * nchan];
    let stride = size.0 * nchan;

    // stage 2+: frames
    let mut cur = d1;
    while cur.has_more_frames() {
        // frame header
        let mut d2 = loop {
            let mut inp = mkinput!();
            let t = Instant::now();
            let r = cur.process(&mut inp, runner!()).unwrap();
            let e = t.elapsed();
            max_call = max_call.max(e);
            calls += 1;
            pos = inp.pos;
            timeline.push((calls, pos, e.as_secs_f64()*1e3, "hdr"));
            match r {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    assert!(pos < data.len(), "truncated at frame header");
                    cur = fallback;
                }
            }
        };
        // frame body
        cur = loop {
            let mut inp = mkinput!();
            let mut obuf = [JxlOutputBuffer::new(&mut buf, size.1, stride)];
            let t = Instant::now();
            let r = d2.process(&mut inp, &mut obuf, runner!()).unwrap();
            let e = t.elapsed();
            max_call = max_call.max(e);
            calls += 1;
            pos = inp.pos;
            timeline.push((calls, pos, e.as_secs_f64()*1e3, "body"));
            match r {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    assert!(pos < data.len(), "truncated in frame");
                    d2 = fallback;
                }
            }
        };
    }
    if trace {
        let mut v = timeline.clone();
        v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        println!("  slowest calls (call#, byte_pos, ms, stage) of {}:", timeline.len());
        for e in v.iter().take(10) {
            println!("    #{:<6} pos={:<10} {:>8.2} ms  {}", e.0, e.1, e.2, e.3);
        }
        let sum: f64 = timeline.iter().map(|x| x.2).sum();
        let over1: f64 = timeline.iter().filter(|x| x.2 > 1.0).map(|x| x.2).sum();
        println!("  total in-call ms={sum:.1}, ms in calls >1ms={over1:.1}");
    }
    if trace {
        println!("  runner.run() invocations={} task closure invocations={}",
            RUN_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            TASK_CALLS.load(std::sync::atomic::Ordering::Relaxed));
    }
    (t0.elapsed(), max_call, calls, size)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let threads = args.get(2).map(|s| s == "mt").unwrap_or(false);
    let trace_on = std::env::var_os("QUICK").is_none();
    println!("file: {path}  runner: {}", if threads { "rayon" } else { "sequential" });
    println!("{:>12} {:>10} {:>12} {:>8}  {}", "chunk", "total ms", "max call ms", "calls", "size");
    let grid: Vec<(Option<usize>, bool)> = if std::env::var_os("QUICK").is_some() {
        vec![(None, false), (Some(4096), false)]
    } else {
        vec![(None, false), (Some(1 << 20), false), (Some(1 << 16), false), (Some(4096), false), (Some(512), false), (Some(1 << 16), true), (Some(4096), true)]
    };
    for (chunk, honest) in grid {
        // warm
        let _ = run(path, chunk, threads, false, honest);
        let (tot, maxc, calls, size) = run(path, chunk, threads, trace_on && chunk == Some(4096), honest);
        let label = match chunk {
            None => "unlimited".to_string(),
            Some(c) => format!("{c}{}", if honest { "*" } else { "" }),
        };
        println!(
            "{:>12} {:>10.1} {:>12.1} {:>8}  {}x{}",
            label,
            tot.as_secs_f64() * 1e3,
            maxc.as_secs_f64() * 1e3,
            calls,
            size.0,
            size.1
        );
    }
}
