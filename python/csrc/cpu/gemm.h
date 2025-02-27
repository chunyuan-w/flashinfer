#pragma once

#include <ATen/native/CPUBlas.h>

// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 2
#define TILE_SIZE 512

// block size for AMX gemm
constexpr int block_size_m() { return 1 * TILE_M; }
constexpr int block_size_n() { return 4 * TILE_N; }

// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

// dispatch: bfloat16, float16, float8_e4m3
#define CPU_DISPATCH_FLOAT_TYPES(TYPE, ...)                                     \
  [&] {                                                                         \
    switch (TYPE) {                                                             \
      case at::ScalarType::BFloat16 : {                                         \
        using scalar_t = at::BFloat16;                                          \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::ScalarType::Half: {                                              \
        using scalar_t = at::Half;                                              \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::ScalarType::Float8_e4m3fn : {                                    \
        using scalar_t = at::Float8_e4m3fn;                                     \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      default:                                                                  \
        TORCH_CHECK(false, "Unsupported floating data type.\n");                \
    }                                                                           \
  }()

#define CPU_DISPATCH_PACKED_FLOAT_TYPES(TYPE1, TYPE2, ...)                      \
  [&] {                                                                         \
    switch (TYPE2) {                                                            \
      case at::ScalarType::Float8_e4m3fn : {                                    \
        TORCH_CHECK(TYPE1 == at::kBFloat16);                                    \
        using scalar_t = at::BFloat16;                                          \
        using packed_t = at::Float8_e4m3fn;                                     \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::ScalarType::BFloat16 : {                                         \
        TORCH_CHECK(TYPE1 == at::kBFloat16);                                    \
        using scalar_t = at::BFloat16;                                          \
        using packed_t = at::BFloat16;                                          \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::ScalarType::Half: {                                              \
        TORCH_CHECK(TYPE1 == at::kHalf);                                        \
        using scalar_t = at::Half;                                              \
        using packed_t = at::Half;                                              \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      default:                                                                  \
        TORCH_CHECK(false, "Unsupported floating data type for weight.\n");     \
    }                                                                           \
  }()

inline void check_scalar_types(at::ScalarType st1, at::ScalarType st2, bool use_fp8_w8a16) {
  if (use_fp8_w8a16) {
    TORCH_CHECK(st1 == at::kBFloat16 && st2 == at::kFloat8_e4m3fn, "Only support bfloat16 with float8_e4m3fn.");
  } else {
    TORCH_CHECK(st1 == st2 && (st1 == at::kBFloat16 || st1 == at::kHalf), "Expect mat1 and mat2 to be bfloat16 or half.")
  }
}

// pack weight to vnni format
at::Tensor convert_weight_packed(at::Tensor& weight);
