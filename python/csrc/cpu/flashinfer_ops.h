#pragma once

#include <ATen/ATen.h>

// layernorm
void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
    int64_t cuda_stream);

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
    int64_t cuda_stream);

// activation
void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

// attention
void decode_attention(at::Tensor& query, at::Tensor& output,
    at::Tensor& k_cache, at::Tensor& v_cahce, at::Tensor& attn_logits,
    at::Tensor& req_to_token, at::Tensor& req_pool_indices,
    at::Tensor& seq_lens, double scaling, double logit_cap);
