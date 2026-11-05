#![forbid(unsafe_code)]

use zenjxl_decoder_simd::{F32SimdVec, I32SimdVec, SimdDescriptor};

fn check<D: SimdDescriptor>(modern: D, legacy: D) {
    for seed in 0..17 {
        let input: [f32; 16] = core::array::from_fn(|i| (i as f32 - seed as f32) * 0.125);
        let evaluate = |d| {
            let a = D::F32Vec::load(d, &input);
            let b = D::F32Vec::splat(d, 0.75);
            let c = a.mul_add(b, a + b);
            let mut output = [0.0; 16];
            c.store(&mut output);
            let mut integers = [0; 16];
            c.bitcast_to_i32().store(&mut integers);
            let mut bytes = [0; 16];
            c.round_store_u8(&mut bytes);
            let mut words = [0; 16];
            c.round_store_u16(&mut words);
            (output.map(f32::to_bits), integers, bytes, words)
        };
        assert_eq!(evaluate(modern), evaluate(legacy));
    }
}

#[cfg(any(
    all(target_arch = "x86_64", feature = "sse42"),
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", feature = "wasm128"),
))]
macro_rules! tier {
    ($name:ident, $descriptor:ident, $token:ident) => {
        #[test]
        fn $name() {
            use archmage::SimdToken;
            #[archmage::arcane]
            #[allow(deprecated)] // Verify the retained compatibility constructors.
            fn run(token: archmage::$token) {
                let modern = zenjxl_decoder_simd::target_features::$name();
                let legacy = zenjxl_decoder_simd::$descriptor::from_token(token);
                check(modern, legacy);
                check(zenjxl_decoder_simd::$descriptor::from_context(), legacy);
            }
            if let Some(token) = archmage::$token::summon() {
                run(token);
            } else {
                assert_ne!(
                    std::env::var("JXL_REQUIRE_TOKEN").as_deref(),
                    Ok(stringify!($token)),
                    "required SIMD tier unavailable"
                );
            }
        }
    };
}

#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
tier!(descriptor_v2, Sse42Descriptor, X64V2Token);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
tier!(descriptor_v3, AvxDescriptor, X64V3Token);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
tier!(descriptor_v4, Avx512Descriptor, X64V4Token);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
tier!(descriptor_neon, NeonDescriptor, NeonToken);
#[cfg(all(target_arch = "wasm32", feature = "wasm128"))]
tier!(descriptor_wasm128, Wasm128Descriptor, Wasm128Token);

#[test]
fn scalar_unit_constructor_matches_legacy() {
    #[allow(deprecated)]
    let legacy = zenjxl_decoder_simd::ScalarDescriptor::from_token(archmage::ScalarToken);
    check(zenjxl_decoder_simd::ScalarDescriptor, legacy);
}

#[cfg(all(target_arch = "wasm32", feature = "wasm128"))]
#[test]
fn wasm_baseline_safe_composition() {
    let descriptor = zenjxl_decoder_simd::target_features::descriptor_wasm128();
    let context = zenjxl_decoder_simd::Wasm128Descriptor::from_context();
    check(descriptor, context);
}
