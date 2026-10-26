/* Test-only independent local layer samples. Uses libjxl's public decoder API.
 * Derived from zencodec perf/spatial-validation/jxl_oracle.c at ea6544561097.
 * No codec implementation or encoded fixture is authored here. */
#include <inttypes.h>
#include <jxl/decode.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int save_plane(const char *directory, unsigned layer, unsigned plane,
                      const float *samples, size_t count, unsigned stride,
                      unsigned channel) {
  char path[4096];
  int n = snprintf(path, sizeof(path), "%s/layer-%u-%u.f32le", directory, layer, plane);
  if (n < 0 || (size_t)n >= sizeof(path)) return 0;
  FILE *file = fopen(path, "wbx");
  if (!file) return 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    memcpy(&bits, samples + i * stride + channel, sizeof(bits));
    unsigned char bytes[4] = {bits & 255, (bits >> 8) & 255,
                             (bits >> 16) & 255, bits >> 24};
    if (fwrite(bytes, 1, 4, file) != 4) { fclose(file); return 0; }
  }
  return fclose(file) == 0;
}

static void blend(const JxlBlendInfo *info) {
  printf("{\"mode\":%u,\"source\":%u,\"alpha\":%u,\"clamp\":%u}",
         info->blendmode, info->source, info->alpha, info->clamp);
}

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  FILE *file = fopen(argv[1], "rb");
  if (!file || fseek(file, 0, SEEK_END)) return 3;
  long length = ftell(file);
  if (length < 1 || length > 64 * 1024 * 1024 || fseek(file, 0, SEEK_SET)) return 4;
  unsigned char *bytes = malloc((size_t)length);
  if (!bytes || fread(bytes, 1, (size_t)length, file) != (size_t)length) return 5;
  fclose(file);
  JxlDecoder *decoder = JxlDecoderCreate(NULL);
  if (!decoder) return 6;
  void *planes[17] = {0};
  size_t sizes[17] = {0};
  JxlBasicInfo basic = {0};
  JxlFrameHeader header = {0};
  unsigned count = 0;
  int rc = 7;
  if (JxlDecoderSetCoalescing(decoder, JXL_FALSE) != JXL_DEC_SUCCESS ||
      JxlDecoderSetKeepOrientation(decoder, JXL_TRUE) != JXL_DEC_SUCCESS ||
      JxlDecoderSetRenderSpotcolors(decoder, JXL_FALSE) != JXL_DEC_SUCCESS ||
      JxlDecoderSubscribeEvents(decoder, JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                                JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS ||
      JxlDecoderSetInput(decoder, bytes, (size_t)length) != JXL_DEC_SUCCESS) goto end;
  JxlDecoderCloseInput(decoder);
  printf("{\"libjxl\":%u,\"layers\":[", JxlDecoderVersion());
  for (unsigned events = 0; events < 4096; ++events) {
    JxlDecoderStatus status = JxlDecoderProcessInput(decoder);
    if (status == JXL_DEC_SUCCESS) { if (count) { printf("]}\n"); rc = 0; } break; }
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(decoder, &basic) != JXL_DEC_SUCCESS ||
          basic.num_color_channels != 3 || basic.num_extra_channels > 16) goto end;
    } else if (status == JXL_DEC_FRAME) {
      if (JxlDecoderGetFrameHeader(decoder, &header) != JXL_DEC_SUCCESS) goto end;
      printf("%s{\"origin\":[%d,%d],\"size\":[%u,%u],\"blend\":",
             count ? "," : "", header.layer_info.crop_x0, header.layer_info.crop_y0,
             header.layer_info.xsize, header.layer_info.ysize);
      blend(&header.layer_info.blend_info);
      printf(",\"extra_blends\":[");
      for (unsigned i = 0; i < basic.num_extra_channels; ++i) {
        JxlBlendInfo info;
        if (JxlDecoderGetExtraChannelBlendInfo(decoder, i, &info) != JXL_DEC_SUCCESS) goto end;
        if (i) printf(",");
        blend(&info);
      }
      printf("],\"save_as_reference\":%u,\"duration\":%u,\"is_last\":%u,\"planes\":[",
             header.layer_info.save_as_reference, header.duration, header.is_last);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
      size_t total = 0;
      for (unsigned i = 0; i <= basic.num_extra_channels; ++i) {
        JxlDecoderStatus queried = i == 0
          ? JxlDecoderImageOutBufferSize(decoder, &format, &sizes[i])
          : JxlDecoderExtraChannelBufferSize(decoder, &format, &sizes[i], i - 1);
        if (queried != JXL_DEC_SUCCESS || sizes[i] > 64 * 1024 * 1024 ||
            total + sizes[i] > 128 * 1024 * 1024) goto end;
        total += sizes[i];
        free(planes[i]);
        planes[i] = malloc(sizes[i]);
        if (!planes[i]) goto end;
        JxlDecoderStatus assigned = i == 0
          ? JxlDecoderSetImageOutBuffer(decoder, &format, planes[i], sizes[i])
          : JxlDecoderSetExtraChannelBuffer(decoder, &format, planes[i], sizes[i], i - 1);
        if (assigned != JXL_DEC_SUCCESS) goto end;
        format.num_channels = 1;
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      size_t pixels = (size_t)header.layer_info.xsize * header.layer_info.ysize;
      for (unsigned i = 0; i < 3 + basic.num_extra_channels; ++i) {
        unsigned p = i < 3 ? 0 : i - 2;
        unsigned stride = i < 3 ? 3 : 1;
        unsigned channel = i < 3 ? i : 0;
        if (sizes[p] != pixels * stride * sizeof(float) ||
            !save_plane(argv[2], count, i, planes[p], pixels, stride, channel)) goto end;
        printf("%s\"layer-%u-%u.f32le\"", i ? "," : "", count, i);
      }
      printf("]}");
      ++count;
    } else { fprintf(stderr, "unexpected decoder status %d\n", status); break; }
  }
end:
  JxlDecoderDestroy(decoder);
  for (unsigned i = 0; i < 17; ++i) free(planes[i]);
  free(bytes);
  return rc;
}
