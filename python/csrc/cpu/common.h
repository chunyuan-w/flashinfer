#pragma once

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
//#include <ATen/native/CPUBlas.h>
//#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

namespace {

inline void check_shape(const at::Tensor& a, const at::Tensor& b, const char* a_name,
                        const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ", a.dim(), " vs ",
              b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name, ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define TORCH_UNUSED(x) (void)(x)

#define CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads)                   \
  TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "num_qo_heads(", num_qo_heads, \
              ") must be divisible by num_kv_heads(", num_kv_heads, ")")

#define CHECK_CPU(x) TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimention")

#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CPU(x);                            \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

inline bool is_float8_tensor(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Float8_e4m3fn ||
         tensor.scalar_type() == at::ScalarType::Float8_e5m2;
}

// TODO: add native dtype conversion instrincs here
//
//

} // anonymous namespace
