// Copyright (c) Imazen LLC. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

// Observe unmodified libjxl v0.12 HLG OOTF around its unity deadband.
#include <cstdio>
#include "lib/jxl/cms/tone_mapping.h"
struct ObservedHlg : jxl::HlgOOTF_Base {
  ObservedHlg(float exponent, const jxl::Vector3& weights)
      : jxl::HlgOOTF_Base(1.0f + exponent, weights) {}
  float exponent() const { return exponent_; }
};
int main() {
  for (float requested : {-.1f, -.099f, -.02f, -.010001f, -.01f, -.009999f,
                          0.f, .009999f, .01f, .010001f, .02f, .099f, .1f}) {
    for (const auto& weights : {jxl::Vector3{.2126f,.7152f,.0722f},
                                jxl::Vector3{.2627f,.6780f,.0593f}}) {
      const ObservedHlg observer(requested, weights);
      for (const auto& input : {jxl::Color{.125f,.25f,.5f},
                                jxl::Color{.001f,.002f,.003f},
                                jxl::Color{.9f,.8f,.7f}}) {
        auto output = input;
        observer.Apply(output);
        std::printf("%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\n",
          observer.exponent(), weights[0],weights[1],weights[2],input[0],input[1],input[2],output[0],output[1],output[2]);
      }
    }
  }
}
