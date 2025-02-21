#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t, typename func_t, typename vec_func_t>
void act_and_mul_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int num_tokens, int dim,
    const func_t& f,
    const vec_func_t& vf) {

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int kVecSize = bVec::size();
  at::parallel_for(0, num_tokens, 0, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      // local ptrs
      const scalar_t* __restrict__ input_ptr = input + i * 2 * dim;
      const scalar_t* __restrict__ input_other_ptr = input_ptr + dim;
      scalar_t* __restrict__ output_ptr = output + i * dim;

      int d;
      for (d = 0; d <= dim - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        bVec y_bvec = bVec::loadu(input_other_ptr + d);
        fVec y_fvec0, y_fvec1;
        std::tie(y_fvec0, y_fvec1) = at::vec::convert_to_float(y_bvec);

        x_fvec0 = vf(x_fvec0);
        x_fvec1 = vf(x_fvec1);

        x_fvec0 = x_fvec0 * y_fvec0;
        x_fvec1 = x_fvec1 * y_fvec1;

        x_bvec = at::vec::convert_from_float<scalar_t>(x_fvec0, x_fvec1);
        x_bvec.store(output_ptr + d);
      }
      for (; d < dim; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        float y_val = static_cast<float>(input_other_ptr[d]);
        output_ptr[d] = f(x_val) * y_val;
      }
    }
  });
}

} // anonymous namespace

// input   : {num_tokens, 2 * d}
// output  : {num_tokens, d}
void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
  int d = input.size(-1) / 2;
  int num_tokens = input.numel() / input.size(-1);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [](float x) { return x / (1.f + std::exp(-x)); },
        [](Vec x) { return x / (Vec(1.f) + x.neg().exp()); });
  });

  TORCH_UNUSED(cuda_stream);
}
