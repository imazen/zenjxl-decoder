// Native v0.12 lossless HLG source for the scene-to-display decoder path.
#include <filesystem>
#include <fstream>
#include <vector>
#include "jxl/encode.h"
#include "jxl/color_encoding.h"
int main(int argc,char** argv) {
  if(argc!=2 || JxlEncoderVersion()!=12000 || std::filesystem::exists(argv[1]))return 2;
  JxlEncoder* encoder=JxlEncoderCreate(nullptr);
  auto ok=[](JxlEncoderStatus s){return s==JXL_ENC_SUCCESS;};
  JxlBasicInfo info;JxlEncoderInitBasicInfo(&info);
  info.xsize=9;info.ysize=3;info.bits_per_sample=32;info.exponent_bits_per_sample=8;
  info.num_color_channels=3;info.uses_original_profile=JXL_TRUE;info.intensity_target=255;
  if(!ok(JxlEncoderSetCodestreamLevel(encoder,10))||!ok(JxlEncoderSetBasicInfo(encoder,&info)))return 3;
  JxlColorEncoding color;JxlColorEncodingSetToSRGB(&color,JXL_FALSE);
  color.transfer_function=JXL_TRANSFER_FUNCTION_HLG;color.rendering_intent=JXL_RENDERING_INTENT_RELATIVE;
  if(!ok(JxlEncoderSetColorEncoding(encoder,&color)))return 4;
  auto* settings=JxlEncoderFrameSettingsCreate(encoder,nullptr);
  if(!ok(JxlEncoderSetFrameLossless(settings,JXL_TRUE))||!ok(JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_EFFORT,3)))return 5;
  std::vector<float> input(9*3*3);const float sample[]{.125f,.25f,.5f};
  for(size_t i=0;i<input.size();++i)input[i]=sample[i%3];
  JxlPixelFormat format{3,JXL_TYPE_FLOAT,JXL_NATIVE_ENDIAN,0};
  if(!ok(JxlEncoderAddImageFrame(settings,&format,input.data(),input.size()*sizeof(float))))return 6;
  JxlEncoderCloseInput(encoder);std::vector<unsigned char> bytes(8192);size_t used=0;
  for(;;){auto* next=bytes.data()+used;size_t available=bytes.size()-used;
    const auto status=JxlEncoderProcessOutput(encoder,&next,&available);used=bytes.size()-available;
    if(status==JXL_ENC_SUCCESS)break;if(status!=JXL_ENC_NEED_MORE_OUTPUT)return 7;bytes.resize(bytes.size()*2);
  }
  JxlEncoderDestroy(encoder);std::ofstream out(argv[1],std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),used);return out?0:8;
}
