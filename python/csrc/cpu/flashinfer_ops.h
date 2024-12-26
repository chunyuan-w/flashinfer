#pragma once

#include <ATen/ATen.h>

void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
    int64_t cuda_stream);

void gemma_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
    int64_t cuda_stream);

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
    int64_t cuda_stream);

void gemma_fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
    int64_t cuda_stream);

void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

void gelu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);
