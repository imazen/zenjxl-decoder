//! Tests demonstrating the property_ranges memory amplification in MA tree decoding.
//!
//! See SECURITY-VULNERABILITY-REPORT.md for full details.

/// Find the maximum property count across all conformance test images.
/// This helps establish a reasonable limit for property_ranges.
#[test]
fn test_find_max_properties() {
    use crate::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use crate::image::{Image, Rect};

    let test_dir = "/home/lilith/work/jxl-rs/jxl/resources/test/conformance_test_images";
    let paths: Vec<_> = std::fs::read_dir(test_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "jxl").unwrap_or(false))
        .collect();

    eprintln!(
        "\n=== Scanning {} conformance images for max properties ===\n",
        paths.len()
    );

    for path in &paths {
        let filename = path.file_name().unwrap().to_string_lossy();

        // Skip animation files which may cause panics
        if filename.contains("animation") || filename.contains("blendmodes") {
            eprintln!("SKIP {} (animation/blend)", filename);
            continue;
        }

        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP {}: read error {:?}", filename, e);
                continue;
            }
        };
        eprint!("Processing {}... ", filename);

        let mut input = data.as_slice();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);

        // Read header
        let mut decoder = match (|| -> Result<_, ()> {
            loop {
                match decoder.process(&mut input) {
                    Ok(ProcessingResult::Complete { result }) => return Ok(result),
                    Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
                    Err(_) => return Err(()),
                }
            }
        })() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("header error");
                continue;
            }
        };

        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;

        let format = JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        };
        decoder.set_pixel_format(format);
        let channels = 3usize;

        // Get frame info
        let mut decoder = match (|| -> Result<_, ()> {
            loop {
                match decoder.process(&mut input) {
                    Ok(ProcessingResult::Complete { result }) => return Ok(result),
                    Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                        if input.is_empty() {
                            return Err(());
                        }
                        decoder = fallback;
                    }
                    Err(_) => return Err(()),
                }
            }
        })() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("frame info error");
                continue;
            }
        };

        let mut output = match Image::<f32>::new((width * channels, height)) {
            Ok(o) => o,
            Err(_) => continue,
        };
        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            output
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * channels, height),
                })
                .into_raw(),
        )];

        // Decode frame (this triggers Tree::read which prints TREE_STATS)
        // Use catch_unwind to handle panics from unsupported features
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                match decoder.process(&mut input, &mut buffers) {
                    Ok(ProcessingResult::Complete { .. }) => break,
                    Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                        if input.is_empty() {
                            break;
                        }
                        decoder = fallback;
                    }
                    Err(_) => break,
                }
            }
        }));
        if result.is_err() {
            eprintln!("(panicked)");
        } else {
            eprintln!("ok");
        }
    }

    eprintln!("\n=== Scan complete ===");
}

/// Test spot.jxl specifically since it may have higher property counts.
#[test]
fn test_spot_jxl() {
    use crate::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use crate::image::{Image, Rect};

    let path = "/home/lilith/work/jxl-rs/jxl/resources/test/conformance_test_images/spot.jxl";
    let data = std::fs::read(path).unwrap();
    eprintln!("\n=== Testing {} ({} bytes) ===\n", path, data.len());

    let mut input = data.as_slice();
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    // Read header
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => {
                eprintln!("Header error: {:?}", e);
                return;
            }
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    eprintln!("Size: {}x{}", width, height);
    eprintln!("Extra channels: {:?}", basic_info.extra_channels);

    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);
    let channels = 3usize;

    // Get frame info
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    eprintln!("Unexpected end of input");
                    return;
                }
                decoder = fallback;
            }
            Err(e) => {
                eprintln!("Frame info error: {:?}", e);
                return;
            }
        }
    };

    let mut output = Image::<f32>::new((width * channels, height)).unwrap();
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    // Decode frame
    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => {
                eprintln!("Decode complete");
                break;
            }
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    eprintln!("Unexpected end of input during frame");
                    break;
                }
                decoder = fallback;
            }
            Err(e) => {
                eprintln!("Decode error: {:?}", e);
                break;
            }
        }
    }
}

/// Test explaining why crafting a malicious tree bitstream is complex.
#[test]
fn test_craft_notes() {
    eprintln!("=== Crafted Tree Notes ===");
    eprintln!("Crafting a full malicious tree bitstream is complex due to entropy coding.");
    eprintln!("The issue is proven by code analysis:");
    eprintln!(
        "  tree.rs:364: property_ranges = Vec::new_with_capacity(num_properties * tree.len())"
    );
    eprintln!("  - tree.len() can be up to tree_size_limit (4M for 16MP images)");
    eprintln!("  - num_properties = max_property + 1, where max_property can be 0-255");
    eprintln!("  - Result: up to 256 * 4M = 1 billion entries = 8 GB allocation");
    eprintln!();
    eprintln!("To craft a malicious file, one would need to:");
    eprintln!("1. Use libjxl encoder to create a base modular file");
    eprintln!("2. Binary-patch the entropy-coded tree section");
    eprintln!("This requires understanding the exact ANS/Huffman state.");
}

/// Demonstrates the property_ranges amplification math.
#[test]
fn test_property_ranges_amplification() {
    let tree_nodes = 1000usize;
    let max_property = 255u8;
    let num_properties = max_property as usize + 1;
    let product = num_properties * tree_nodes;
    let bytes = product * 8; // (i32, i32) tuple

    eprintln!("=== Property Ranges Amplification Demo ===");
    eprintln!("Tree nodes: {}", tree_nodes);
    eprintln!(
        "Max property: {} -> num_properties: {}",
        max_property, num_properties
    );
    eprintln!(
        "product: {} entries = {} bytes ({:.2} MB)",
        product,
        bytes,
        bytes as f64 / 1024.0 / 1024.0
    );

    assert!(
        bytes > 1_000_000,
        "Even 1000 nodes with max properties is >1MB"
    );
}

/// Theoretical worst case for a 16MP image.
#[test]
fn test_theoretical_attack_size() {
    let image_size: usize = 4096 * 4096; // 16MP
    let channels = 3;

    // tree_size_limit from decode.rs:230-235
    let tree_size_limit = (1024 + image_size * channels / 16).min(1 << 22);

    // Attacker-controlled: max_property = 255 -> num_properties = 256
    let max_properties = 256;

    // Attack allocation
    let property_ranges_entries = max_properties * tree_size_limit;
    let property_ranges_bytes = property_ranges_entries * 8; // (i32, i32)

    // Normal output size
    let output_bytes = image_size * channels * 4; // f32 RGB

    let amplification = property_ranges_bytes as f64 / output_bytes as f64;

    eprintln!("=== Theoretical Attack for {}x{} image ===", 4096, 4096);
    eprintln!("tree_size_limit: {} nodes", tree_size_limit);
    eprintln!("max_properties: {}", max_properties);
    eprintln!(
        "property_ranges: {} entries = {} bytes ({:.1} GB)",
        property_ranges_entries,
        property_ranges_bytes,
        property_ranges_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "output_bytes: {} ({:.1} MB)",
        output_bytes,
        output_bytes as f64 / 1024.0 / 1024.0
    );
    eprintln!("amplification: {:.1}x", amplification);

    // Verify the attack is significant
    assert!(
        property_ranges_bytes > 1024 * 1024 * 1024,
        "Attack should cause >1GB allocation"
    );
    assert!(amplification > 10.0, "Amplification should be >10x");
}
