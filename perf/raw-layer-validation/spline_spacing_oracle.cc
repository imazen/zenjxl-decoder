// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test-only point-walking oracle, preserving the floating-point operation order
// of ForEachEquallySpacedPoint in pinned libjxl v0.12:
// https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/splines.cc
// Compile without floating-point contraction; the reference's common scalar
// point walker is outside its per-target SIMD rendering namespaces.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

struct Point { float x, y; };
Point operator-(Point a, Point b) { return {a.x - b.x, a.y - b.y}; }
Point operator+(Point a, Point b) { return {a.x + b.x, a.y + b.y}; }
Point operator*(float k, Point p) { return {k * p.x, k * p.y}; }
uint32_t bits(float value) { uint32_t result; std::memcpy(&result, &value, 4); return result; }

template <class F>
void Walk(const std::vector<Point>& points, const F& emit) {
  Point current = points.front();
  emit(current, 1.0f);
  auto next = points.begin();
  while (next != points.end()) {
    const Point* previous = &current;
    float distance_from_previous = 0.0f;
    for (;;) {
      if (next == points.end()) { emit(*previous, distance_from_previous); return; }
      const Point delta = *next - *previous;
      const float distance_to_next = std::sqrt(delta.x * delta.x + delta.y * delta.y);
      if (distance_from_previous + distance_to_next >= 1.0f) {
        current = *previous + ((1.0f - distance_from_previous) / distance_to_next) * delta;
        emit(current, 1.0f);
        break;
      }
      distance_from_previous += distance_to_next;
      previous = &*next;
      ++next;
    }
  }
}

void Print(const char* name, std::initializer_list<Point> points) {
  std::printf("const %s: &[[u32; 3]] = &[\n", name);
  Walk(std::vector<Point>(points), [](Point point, float weight) {
    std::printf("    [0x%08x, 0x%08x, 0x%08x],\n", bits(point.x), bits(point.y), bits(weight));
  });
  std::printf("];\n");
}
int main() {
  Print("SPACING_CURVE", {{-2.75f, 1.25f}, {1.1f, 0.7f}, {2.4f, 3.9f}, {10.2f, -0.35f}, {13.1f, 7.6f}});
  Print("SPACING_SINGLE", {{0.25f, -0.75f}});
  Print("SPACING_REPEATED", {{0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}});
}
