import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compare

from flashinfer.activation import silu_and_mul, gelu_and_mul, gelu_tanh_and_mul

torch.manual_seed(1111)

# def forward_native(x: torch.Tensor) -> torch.Tensor:
#     d = x.shape[-1] // 2
#     return F.silu(x[..., :d]) * x[..., d:]

def forward_native(x: torch.Tensor, approximate="tanh") -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]

#def forward_native(x: torch.Tensor) -> torch.Tensor:
#    orig_dtype = x.dtype
#    x = x.to(torch.float32)
#    d = x.shape[-1] // 2
#    return (F.silu(x[..., :d]) * x[..., d:]).to(orig_dtype)

def run_single_test(shape, dtype, device="cuda"):
    x = torch.randn(shape, dtype=dtype).to(device=device)


    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # silu_and_mul(x, out)
    # gelu_and_mul(x, out)
    gelu_tanh_and_mul(x, out)

    ref_out = forward_native(x)

    compare(out, ref_out, True)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)


run_single_test([128, 22016], torch.bfloat16, "cpu")
run_single_test([129, 22016], torch.float16, "cpu")
