// Copyright (c) Imazen LLC. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.
// Public-API observer/producer for pinned libjxl v0.12. Test tooling only.
#include "jxl/color_encoding.h"
#include "jxl/decode.h"
#include "jxl/encode.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
void enc_ok(JxlEncoderStatus s) {
  if (s != JXL_ENC_SUCCESS)
    throw std::runtime_error("encoder refused control");
}
void dec_ok(JxlDecoderStatus s) {
  if (s != JXL_DEC_SUCCESS)
    throw std::runtime_error("decoder API refused control");
}
void save(const fs::path &p, const void *data, size_t size) {
  if (fs::exists(p))
    throw std::runtime_error("refusing to overwrite evidence");
  std::ofstream f(p, std::ios::binary);
  f.write(static_cast<const char *>(data), size);
  if (!f)
    throw std::runtime_error("write failed");
}
void produce(const fs::path &path, const std::vector<uint32_t> &times,
             float solidity, bool spot, bool layer,
             const std::vector<uint8_t> *cmyk_icc = nullptr, int level = 10,
             const std::vector<uint32_t> &durations = {}) {
  JxlEncoder *e = JxlEncoderCreate(nullptr);
  if (!e)
    throw std::runtime_error("encoder allocation");
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 4;
  info.ysize = 3;
  info.bits_per_sample = 8;
  info.uses_original_profile = JXL_TRUE;
  info.num_extra_channels = spot || cmyk_icc ? 1 : 0;
  info.have_animation = !times.empty();
  info.animation.have_timecodes = !times.empty();
  info.animation.tps_numerator = 100;
  info.animation.tps_denominator = 1;
  enc_ok(JxlEncoderSetCodestreamLevel(e, level));
  enc_ok(JxlEncoderSetBasicInfo(e, &info));
  if (spot || cmyk_icc) {
    JxlExtraChannelInfo ec;
    JxlEncoderInitExtraChannelInfo(
        spot ? JXL_CHANNEL_SPOT_COLOR : JXL_CHANNEL_BLACK, &ec);
    ec.bits_per_sample = 8;
    ec.spot_color[0] = .25f;
    ec.spot_color[1] = .5f;
    ec.spot_color[2] = .75f;
    ec.spot_color[3] = solidity;
    enc_ok(JxlEncoderSetExtraChannelInfo(e, 0, &ec));
  }
  if (cmyk_icc) {
    enc_ok(JxlEncoderSetICCProfile(e, cmyk_icc->data(), cmyk_icc->size()));
  } else {
    JxlColorEncoding color;
    JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
    enc_ok(JxlEncoderSetColorEncoding(e, &color));
  }
  auto *settings = JxlEncoderFrameSettingsCreate(e, nullptr);
  enc_ok(JxlEncoderSetFrameLossless(settings, JXL_TRUE));
  enc_ok(JxlEncoderFrameSettingsSetOption(settings,
                                          JXL_ENC_FRAME_SETTING_EFFORT, 3));
  enc_ok(JxlEncoderFrameSettingsSetOption(
      settings, JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE, 0));
  const JxlPixelFormat rgb{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat one{1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  for (size_t frame = 0; frame < (times.empty() ? 1 : times.size()); ++frame) {
    JxlFrameHeader h;
    JxlEncoderInitFrameHeader(&h);
    h.duration = durations.empty() ? 1 : durations.at(frame);
    h.timecode = times.empty() ? 0 : times[frame];
    if (layer && frame == 1) {
      h.layer_info.have_crop = JXL_TRUE;
      h.layer_info.crop_x0 = -1;
      h.layer_info.crop_y0 = 1;
      h.layer_info.xsize = 3;
      h.layer_info.ysize = 2;
      h.layer_info.blend_info.blendmode = JXL_BLEND_ADD;
    }
    enc_ok(JxlEncoderSetFrameHeader(settings, &h));
    const size_t pixels = layer && frame == 1 ? 6 : 12;
    std::vector<uint8_t> input(pixels * 3), extra(pixels);
    for (size_t i = 0; i < pixels; ++i) {
      input[3 * i] = (i + frame) % 2 ? 255 : 0;
      input[3 * i + 1] = (i + frame) % 3 ? 0 : 255;
      input[3 * i + 2] = frame ? 255 : 0;
      extra[i] = i % 2 ? 255 : 0;
    }
    enc_ok(JxlEncoderAddImageFrame(settings, &rgb, input.data(), input.size()));
    if (spot || cmyk_icc)
      enc_ok(JxlEncoderSetExtraChannelBuffer(settings, &one, extra.data(),
                                             extra.size(), 0));
  }
  JxlEncoderCloseInput(e);
  std::vector<uint8_t> bytes(4096);
  size_t used = 0;
  for (;;) {
    uint8_t *next = bytes.data() + used;
    size_t avail = bytes.size() - used;
    auto s = JxlEncoderProcessOutput(e, &next, &avail);
    used = bytes.size() - avail;
    if (s == JXL_ENC_SUCCESS)
      break;
    if (s != JXL_ENC_NEED_MORE_OUTPUT)
      throw std::runtime_error("encode failed");
    bytes.resize(bytes.size() * 2);
  }
  JxlEncoderDestroy(e);
  save(path, bytes.data(), used);
}
int observe(const fs::path &input, fs::path prefix, bool coalescing, bool spot,
            std::optional<size_t> seek = std::nullopt) {
  std::ifstream f(input, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
  JxlDecoder *d = JxlDecoderCreate(nullptr);
  dec_ok(JxlDecoderSetCoalescing(d, coalescing));
  dec_ok(JxlDecoderSetRenderSpotcolors(d, spot));
  dec_ok(JxlDecoderSetKeepOrientation(d, JXL_TRUE));
  dec_ok(JxlDecoderSetUnpremultiplyAlpha(d, JXL_FALSE));
  dec_ok(JxlDecoderSubscribeEvents(d, JXL_DEC_BASIC_INFO |
                                          JXL_DEC_COLOR_ENCODING |
                                          JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  dec_ok(JxlDecoderSetInput(d, bytes.data(), bytes.size()));
  JxlDecoderCloseInput(d);
  JxlBasicInfo info{};
  JxlFrameHeader frame{};
  std::vector<float> output;
  size_t count = 0;
  JxlPixelFormat format{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (;;) {
    auto s = JxlDecoderProcessInput(d);
    if (s == JXL_DEC_BASIC_INFO) {
      dec_ok(JxlDecoderGetBasicInfo(d, &info));
      std::cout << "info " << info.xsize << ' ' << info.ysize << ' '
                << info.num_extra_channels << '\n';
      for (uint32_t i = 0; i < info.num_extra_channels; ++i) {
        JxlExtraChannelInfo ec{};
        dec_ok(JxlDecoderGetExtraChannelInfo(d, i, &ec));
        std::cout << "extra " << i << ' ' << int(ec.type) << ' '
                  << ec.spot_color[3] << '\n';
      }
    } else if (s == JXL_DEC_COLOR_ENCODING) {
      JxlColorEncoding color{};
      if (JxlDecoderGetColorAsEncodedProfile(d,
                                             JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                             &color) == JXL_DEC_SUCCESS) {
        std::cout << "original color " << int(color.color_space) << ' '
                  << int(color.transfer_function) << '\n';
      }
    } else if (s == JXL_DEC_FRAME) {
      dec_ok(JxlDecoderGetFrameHeader(d, &frame));
      std::cout << "frame " << count << ' ' << frame.duration << ' '
                << frame.timecode << ' ' << frame.layer_info.crop_x0 << ' '
                << frame.layer_info.crop_y0 << ' ' << frame.layer_info.xsize
                << ' ' << frame.layer_info.ysize << ' '
                << int(frame.layer_info.blend_info.blendmode) << '\n';
    } else if (s == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t size;
      dec_ok(JxlDecoderImageOutBufferSize(d, &format, &size));
      output.resize(size / sizeof(float));
      dec_ok(JxlDecoderSetImageOutBuffer(d, &format, output.data(), size));
    } else if (s == JXL_DEC_FULL_IMAGE) {
      save(prefix.string() + "-" + std::to_string(count++) + ".f32",
           output.data(), output.size() * sizeof(float));
    } else if (s == JXL_DEC_SUCCESS || s == JXL_DEC_ERROR ||
               s == JXL_DEC_NEED_MORE_INPUT) {
      std::cout << "result " << int(s) << ' ' << count << '\n';
      if (s == JXL_DEC_SUCCESS && seek) {
        JxlDecoderRewind(d);
        JxlDecoderSkipFrames(d, *seek);
        dec_ok(JxlDecoderSetInput(d, bytes.data(), bytes.size()));
        JxlDecoderCloseInput(d);
        seek.reset();
        prefix += "-seek";
        count = 0;
        continue;
      }
      JxlDecoderDestroy(d);
      return s == JXL_DEC_SUCCESS ? 0 : 1;
    } else
      throw std::runtime_error("unexpected decoder event");
  }
}
int main(int argc, char **argv) {
  try {
    if (JxlEncoderVersion() != 12000 || JxlDecoderVersion() != 12000)
      throw std::runtime_error("requires v0.12.0");
    if (argc == 3 && std::string(argv[1]) == "generate") {
      fs::path out(argv[2]);
      if (!fs::create_directory(out))
        throw std::runtime_error("new output directory required");
      produce(out / "plain-srgb.jxl", {}, 0, false, false, nullptr, 5);
      produce(out / "spot-zero.jxl", {}, 0, true, false);
      produce(out / "spot-half.jxl", {}, .5f, true, false);
      produce(out / "spot-one.jxl", {}, 1, true, false);
      produce(out / "spot-negative.jxl", {}, -.5f, true, false);
      produce(out / "spot-two.jxl", {}, 2, true, false);
      produce(out / "time-increasing.jxl", {1, 2}, 0, false, false);
      produce(out / "time-decreasing.jxl", {2, 1}, 0, false, false);
      produce(out / "time-equal.jxl", {2, 2}, 0, false, false);
      produce(out / "time-zero.jxl", {2, 0, 1}, 0, false, false);
      produce(out / "time-zero-increasing.jxl", {2, 0, 3}, 0, false, false);
      produce(out / "time-hidden.jxl", {2, 5, 3}, 0, false, false, nullptr, 10,
              {1, 0, 1});
      produce(out / "cropped-add.jxl", {1, 2}, 0, false, true);
      return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "observe")
      return observe(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]));
    if (argc == 5 && std::string(argv[1]) == "seek")
      return observe(argv[2], argv[3], true, true, std::stoul(argv[4]));
    if (argc == 5 && std::string(argv[1]) == "cmyk") {
      std::ifstream f(argv[2], std::ios::binary);
      std::vector<uint8_t> icc((std::istreambuf_iterator<char>(f)), {});
      if (icc.empty())
        throw std::runtime_error("missing CMYK profile");
      produce(argv[3], {}, 0, false, false, &icc, std::stoi(argv[4]));
      return 0;
    }
    throw std::runtime_error(
        "generate DIR | observe INPUT PREFIX COALESCE "
        "SPOT | seek INPUT PREFIX INDEX | cmyk ICC OUTPUT LEVEL");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
