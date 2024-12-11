#pragma once

#include <ATen/ATen.h>

void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
    int64_t cuda_stream);

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
    int64_t cuda_stream);

