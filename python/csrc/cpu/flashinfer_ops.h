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

// topk
void grouped_topk(
    at::Tensor& topk_weights, at::Tensor& topk_ids, at::Tensor& hidden_states, at::Tensor& gating_output,
    int64_t topk, bool renormalize, int64_t num_expert_group, int64_t topk_group);

// weight prepack
at::Tensor convert_weight_packed(at::Tensor& weight);

// moe
at::Tensor fused_experts(at::Tensor& hidden_states, at::Tensor& w1, at::Tensor& w2,
    at::Tensor& topk_weights, at::Tensor& topk_ids, bool inplace, bool is_vnni);
