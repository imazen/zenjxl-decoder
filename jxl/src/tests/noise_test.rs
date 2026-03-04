#[test]
#[ignore]
fn test_decode_noise_12mp() {
    use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

    let base = std::env::var("JXL_TEST_OUTPUT_DIR")
        .unwrap_or_else(|_| "/mnt/v/output".into());
    let path = format!("{}/jxl-dos-test/noise_12mp.jxl", base);
    if !std::path::Path::new(path).exists() {
        eprintln!("File not found: {}", path);
        return;
    }

    let data = std::fs::read(path).unwrap();
    eprintln!("File size: {} bytes", data.len());

    let mut input = data.as_slice();
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    // Process header
    loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => {
                let info = result.basic_info();
                eprintln!("Image: {}x{}", info.size.0, info.size.1);
                break;
            }
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
            Err(e) => {
                eprintln!("Error: {:?}", e);
                return;
            }
        }
    }
}
