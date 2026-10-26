/* Test-only independent inspection of fixtures written by jxl-encoder. */
#include <inttypes.h>
#include <jxl/decode.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t bits(float f) {
  uint32_t out;
  _Static_assert(sizeof(out) == sizeof(f), "requires binary32 float");
  memcpy(&out, &f, sizeof(out));
  return out;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  FILE *file = fopen(argv[1], "rb");
  if (!file || fseek(file, 0, SEEK_END)) return 3;
  long size = ftell(file);
  if (size < 1 || size > 64 * 1024 * 1024 || fseek(file, 0, SEEK_SET)) return 4;
  unsigned char *data = malloc((size_t)size);
  if (!data || fread(data, 1, (size_t)size, file) != (size_t)size) return 5;
  fclose(file);
  JxlDecoder *decoder = JxlDecoderCreate(NULL);
  if (!decoder) return 6;
  int rc = 7;
  if (JxlDecoderSetKeepOrientation(decoder, JXL_TRUE) != JXL_DEC_SUCCESS ||
      JxlDecoderSubscribeEvents(decoder, JXL_DEC_BASIC_INFO | JXL_DEC_FRAME) != JXL_DEC_SUCCESS ||
      JxlDecoderSetInput(decoder, data, (size_t)size) != JXL_DEC_SUCCESS) goto end;
  JxlDecoderCloseInput(decoder);
  printf("version %" PRIu32 "\n", JxlDecoderVersion());
  for (unsigned events = 0; events < 4096; ++events) {
    JxlDecoderStatus status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_SUCCESS) { rc = 0; break; }
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info = {0};
      if (JxlDecoderGetBasicInfo(decoder, &info) != JXL_DEC_SUCCESS) break;
      printf("display %" PRIu32 " %" PRIu32 "\n", info.intrinsic_xsize, info.intrinsic_ysize);
      printf("tone %" PRIu32 " %" PRIu32 " %u %" PRIu32 "\n",
             bits(info.intensity_target), bits(info.min_nits),
             (unsigned)info.relative_to_max_display, bits(info.linear_below));
      printf("timecodes %u\n", (unsigned)(info.have_animation && info.animation.have_timecodes));
      for (uint32_t i = 0; i < info.num_extra_channels; ++i) {
        JxlExtraChannelInfo extra = {0};
        if (JxlDecoderGetExtraChannelInfo(decoder, i, &extra) != JXL_DEC_SUCCESS) goto end;
        printf("extra %" PRIu32 " %u %" PRIu32 " %" PRIu32 " %" PRIu32
               " %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 "\n",
               i, (unsigned)extra.type, extra.bits_per_sample,
               extra.exponent_bits_per_sample, extra.dim_shift,
               bits(extra.spot_color[0]), bits(extra.spot_color[1]),
               bits(extra.spot_color[2]), bits(extra.spot_color[3]), extra.cfa_channel);
      }
    } else if (status == JXL_DEC_FRAME) {
      JxlFrameHeader frame = {0};
      if (JxlDecoderGetFrameHeader(decoder, &frame) != JXL_DEC_SUCCESS) break;
      printf("frame %" PRIu32 " %" PRIu32 "\n", frame.duration, frame.timecode);
    } else { fprintf(stderr, "unexpected libjxl status %d\n", status); break; }
  }
end:
  JxlDecoderDestroy(decoder);
  free(data);
  return rc;
}
