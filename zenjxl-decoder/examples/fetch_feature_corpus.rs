//! Provision the corpus used by the feature-test caller.
fn main() {
    let corpus = codec_corpus::Corpus::new().expect("open codec-corpus cache");
    let jxl = corpus.get("jxl").expect("fetch JXL feature corpus");
    println!("CODEC_CORPUS_PATH={}", jxl.parent().unwrap().display());
}
