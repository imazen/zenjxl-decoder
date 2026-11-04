//! Provision the corpus used by the feature-test caller.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let corpus = codec_corpus::Corpus::new().expect("open codec-corpus cache");
    let jxl = corpus.get("jxl").expect("fetch JXL feature corpus");
    println!("CODEC_CORPUS_PATH={}", jxl.parent().unwrap().display());
}

#[cfg(target_arch = "wasm32")]
fn main() -> std::process::ExitCode {
    eprintln!("Fetch the feature corpus on the native host before running WASI tests.");
    std::process::ExitCode::FAILURE
}
