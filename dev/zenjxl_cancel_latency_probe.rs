// How fast does zenjxl-decoder's in-library cooperative cancellation actually stop
// a decode, compared to the upstream "chunk the input" workaround?
use std::sync::Arc;
use std::time::{Duration, Instant};

use almost_enough::Stopper;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&path).unwrap();

    // baseline: uncancelled decode
    let t = Instant::now();
    let img = zenjxl_decoder::api::decode_with(
        &data,
        zenjxl_decoder::api::JxlDecoderOptions::default(),
    )
    .unwrap();
    let full = t.elapsed();
    println!(
        "{}: {} bytes -> {}x{}, full decode {:.1} ms",
        path,
        data.len(),
        img.info.size.0,
        img.info.size.1,
        full.as_secs_f64() * 1e3
    );

    // cancel at various fractions of the decode; measure how long after the
    // cancel fires the call actually returns.
    for frac in [0.0f64, 0.1, 0.25, 0.5, 0.75] {
        let stop = Arc::new(Stopper::new());
        let s2 = stop.clone();
        let delay = full.mul_f64(frac);
        let h = std::thread::spawn(move || {
            std::thread::sleep(delay);
            let t = Instant::now();
            s2.cancel();
            t
        });
        let t0 = Instant::now();
        let r = zenjxl_decoder::api::decode_with(
            &data,
            zenjxl_decoder::api::JxlDecoderOptions::default()
                .with_stop(stop as Arc<dyn enough::Stop>),
        );
        let returned = Instant::now();
        let cancel_at = h.join().unwrap();
        let latency = returned.saturating_duration_since(cancel_at);
        println!(
            "  cancel at {:>4.0}% ({:>6.1} ms): call returned {:>7.1} ms after cancel, total {:>6.1} ms, {}",
            frac * 100.0,
            delay.as_secs_f64() * 1e3,
            latency.as_secs_f64() * 1e3,
            t0.elapsed().as_secs_f64() * 1e3,
            if r.is_err() { "Cancelled" } else { "completed" }
        );
    }
    let _ = Duration::ZERO;
}
