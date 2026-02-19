# Test Gap Analysis: jxl-rs vs libjxl

## Summary

| Metric | libjxl | jxl-rs |
|--------|--------|--------|
| Test files | 45 (decoder-relevant) | 55 |
| Test count | ~330 | ~200 |
| Gap | ~130 missing tests | |

## Critical Gaps by Category

### 1. Decoder API Tests (CRITICAL)
**libjxl: 114 tests | jxl-rs: ~15 tests**

Missing in jxl-rs:
- [ ] `JxlSignatureCheckTest` - signature validation edge cases
- [ ] `BufferSizeTest` - output buffer size calculation
- [ ] `ProcessEmptyInputWithBoxes` - empty input handling
- [ ] `ExtraBytesAfterCompressedStream` - trailing bytes handling
- [ ] `ConcatenatedCompressedStreams` - multiple streams
- [ ] `DCNotGettableTest` - DC image unavailability
- [ ] `AlignTest` - buffer alignment requirements
- [ ] `AnimationTestStreaming` - streaming animation
- [ ] `SkipFrameTest` - frame skipping
- [ ] `SkipFrameWithBlendingTest` - skip with blending
- [ ] `SkipFrameWithAlphaBlendingTest` - skip with alpha blend
- [ ] `OrientedCroppedFrameTest` - cropped frames
- [ ] `InputHandlingTestOneShot` - one-shot input
- [ ] `InputHandlingTestStreaming` - streaming input
- [ ] `FlushTest*` - all flush tests
- [ ] `ProgressiveEventTest` - progressive events
- [ ] `ExtendedBoxSizeTest` - extended box sizes
- [ ] `SpotColorTest` - spot color channels

### 2. Entropy Coding Tests (HIGH)
**libjxl: 20 tests | jxl-rs: 8 tests**

Missing in jxl-rs:
- [ ] `EmptyRoundtrip` - empty ANS stream
- [ ] `RandomStreamRoundtrip*` - various random streams
- [ ] `UintConfigRoundtrip` - uint config handling
- [ ] `TestCheckpointingANS` - ANS checkpointing
- [ ] `TestCheckpointingPrefix` - prefix checkpointing
- [ ] `TestCheckpointingANSLZ77` - LZ77 checkpointing
- [ ] `EstimateTokenCost` - cost estimation

### 3. Modular Mode Tests (HIGH)
**libjxl: 21 tests | jxl-rs: ~5 tests**

Missing in jxl-rs:
- [ ] `RoundtripLosslessGroups128` - large group lossless
- [ ] `LargeGss*` - large group size shuffle tests
- [ ] `RoundtripLosslessCustomWpPermuteRCT` - custom predictor
- [ ] `RoundtripLossy*` - lossy modular tests
- [ ] `RoundtripExtraProperties` - extra MA tree properties
- [ ] `PredictorIntegerOverflow` - overflow edge case
- [ ] `UnsqueezeIntegerOverflow` - unsqueeze overflow

### 4. DCT/Transform Tests (MEDIUM)
**libjxl: 15 tests | jxl-rs: ~10 tests**

Missing in jxl-rs:
- [ ] `TestDctAccuracyShard` - DCT accuracy verification
- [ ] `TestIdctAccuracyShard` - IDCT accuracy verification
- [ ] `TestRectInverse` - rectangular DCT inverse
- [ ] `TestRectTranspose` - rectangular transpose
- [ ] AC strategy roundtrip tests

### 5. Blending Tests (MEDIUM)
**libjxl: 6 tests | jxl-rs: ~9 tests**

jxl-rs actually has good blending coverage. Missing:
- [ ] `Crops` - cropped frame blending (libjxl blending_test.cc)

### 6. Color Management Tests (MEDIUM)
**libjxl: 16 tests | jxl-rs: ~25 tests**

jxl-rs has good coverage here. Missing:
- [ ] `GoldenXYBCube` - golden reference XYB test
- [ ] Some ICC profile edge cases

### 7. Progressive/Streaming Tests (HIGH)
**libjxl: 11 tests | jxl-rs: ~3 tests**

Missing in jxl-rs:
- [ ] `RoundtripSmallPasses` - small passes roundtrip
- [ ] `RoundtripMultiGroupPasses` - multi-group passes
- [ ] `ProgressiveDownsample*` - progressive downsample tests
- [ ] `NonProgressiveDCImage` - non-progressive DC

### 8. Quantization Tests (MEDIUM)
**libjxl: 19 tests | jxl-rs: ~2 tests**

Missing in jxl-rs:
- [ ] Weight matrix tests for each DCT size
- [ ] `QuantizerParams` - parameter handling
- [ ] `BitStreamRoundtripSameQuant` - roundtrip tests
- [ ] `BitStreamRoundtripRandomQuant` - random roundtrip

### 9. Roundtrip/Parity Tests (CRITICAL)
**libjxl: 58 tests | jxl-rs: 0 tests**

jxl-rs has NO parity tests against libjxl output. This is the biggest gap.

Missing entirely:
- [ ] ALL of `jxl_test.cc` tests - verify decoder output matches encoder input
- [ ] ALL of `roundtrip_test.cc` tests - format roundtrip verification

## Priority Order for Porting

### Phase 1: Critical Infrastructure
1. **Parity test framework** - infrastructure to compare against libjxl output
2. **Signature/container tests** - security-relevant parsing tests
3. **Empty/malformed input tests** - robustness tests

### Phase 2: Core Decoder
4. **Streaming input tests** - chunked decoding
5. **Animation streaming tests** - animated JXL streaming
6. **Progressive decoding tests** - pass-by-pass verification

### Phase 3: Feature Coverage
7. **Modular mode tests** - lossless mode edge cases
8. **Entropy coding tests** - ANS/Huffman edge cases
9. **Quantization tests** - weight matrix verification

### Phase 4: Completeness
10. **All remaining decode_test.cc tests**
11. **All remaining jxl_test.cc roundtrip tests**

## Test Porting Strategy

For each test ported from libjxl:

1. **Read the C++ test** - understand what it's testing
2. **Create Rust equivalent** - same logic, Rust idioms
3. **Generate reference data** - use libjxl to create expected outputs
4. **DO NOT WEAKEN TOLERANCES** - if test fails, implementation is wrong
5. **Commit failing tests** - document what needs fixing
6. **Fix implementation separately** - in a different commit

## Files to Create

```
jxl/src/tests/
├── mod.rs                    # Test module organization
├── parity.rs                 # Parity test infrastructure
├── decode_api_tests.rs       # Ported from decode_test.cc
├── modular_tests.rs          # Ported from modular_test.cc
├── entropy_tests.rs          # Ported from ans_test.cc
├── streaming_tests.rs        # Streaming input tests
├── animation_tests.rs        # Animation-specific tests
├── progressive_tests.rs      # Progressive decoding tests
└── reference_data/           # Golden reference outputs from libjxl
    ├── README.md
    └── *.bin                 # Binary reference data
```
