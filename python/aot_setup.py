"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import contextlib
import copy
import itertools
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import warnings
from typing import Iterator, List, Tuple

import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__file__).resolve().parents[1]

sys.path.append(str(root / "python"))

from _aot_build_utils import (
    generate_batch_paged_decode_inst,
    generate_batch_paged_prefill_inst,
    generate_batch_ragged_prefill_inst,
    generate_dispatch_inc,
    generate_single_decode_inst,
    generate_single_prefill_inst,
)

force_cuda = os.environ.get("FLASHINFER_FORCE_CUDA", "0") == "1"
enable_bf16 = os.environ.get("FLASHINFER_ENABLE_BF16", "1") == "1"
enable_fp8 = os.environ.get("FLASHINFER_ENABLE_FP8", "1") == "1"

build_cuda_sources = (torch.cuda.is_available() and (torch_cpp_ext.CUDA_HOME is not None)) or force_cuda

print("Flashinfer build configuration:")
print("FLASHINFER_FORCE_CUDA = ", force_cuda)
print("FLASHINFER_ENABLE_BF16 = ", enable_bf16)
print("FLASHINFER_ENABLE_FP8 = ", enable_fp8)
print("BUILD_CUDA_SOURCES = ", build_cuda_sources)

if enable_bf16:
    torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
if enable_fp8:
    torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8")


def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)


def get_instantiation_cu() -> Tuple[List[str], List[str], List[str]]:
    path = root / "python" / "csrc" / "generated"
    path.mkdir(parents=True, exist_ok=True)

    head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64,128,256").split(",")
    pos_encoding_modes = os.environ.get("FLASHINFER_POS_ENCODING_MODES", "0").split(",")
    allow_fp16_qk_reduction_options = os.environ.get(
        "FLASHINFER_ALLOW_FP16_QK_REDUCTION_OPTIONS", "0"
    ).split(",")
    mask_modes = os.environ.get("FLASHINFER_MASK_MODES", "0,1,2").split(",")

    head_dims = list(map(int, head_dims))
    pos_encoding_modes = list(map(int, pos_encoding_modes))
    allow_fp16_qk_reduction_options = list(map(int, allow_fp16_qk_reduction_options))
    mask_modes = list(map(int, mask_modes))
    # dispatch.inc
    write_if_different(
        path / "dispatch.inc",
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                head_dims=head_dims,
                pos_encoding_modes=pos_encoding_modes,
                allow_fp16_qk_reductions=allow_fp16_qk_reduction_options,
                mask_modes=mask_modes,
            )
        ),
    )

    idtypes = ["i32"]
    prefill_dtypes = ["f16"]
    decode_dtypes = ["f16"]
    fp16_dtypes = ["f16"]
    fp8_dtypes = ["e4m3", "e5m2"]
    if enable_bf16:
        prefill_dtypes.append("bf16")
        decode_dtypes.append("bf16")
        fp16_dtypes.append("bf16")
    if enable_fp8:
        decode_dtypes.extend(fp8_dtypes)

    files_decode = []
    files_prefill = []
    single_decode_uris = []
    # single decode files
    for head_dim, pos_encoding_mode in itertools.product(head_dims, pos_encoding_modes):
        for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
            itertools.product(fp16_dtypes, fp8_dtypes)
        ):
            dtype_out = dtype_q
            fname = f"single_decode_head_{head_dim}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}.cu"
            files_decode.append(str(path / fname))
            content = generate_single_decode_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                dtype_q,
                dtype_kv,
                dtype_out,
            )
            for use_sliding_window in [True, False]:
                for use_logits_soft_cap in [True, False]:
                    single_decode_uris.append(
                        f"single_decode_with_kv_cache_dtype_q_{dtype_q}_"
                        f"dtype_kv_{dtype_kv}_"
                        f"dtype_o_{dtype_out}_"
                        f"head_dim_{head_dim}_"
                        f"posenc_{pos_encoding_mode}_"
                        f"use_swa_{use_sliding_window}_"
                        f"use_logits_cap_{use_logits_soft_cap}"
                    )
            write_if_different(path / fname, content)

    # batch decode files
    batch_decode_uris = []
    for (
        head_dim,
        pos_encoding_mode,
    ) in itertools.product(
        head_dims,
        pos_encoding_modes,
    ):
        for idtype in idtypes:
            for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
                itertools.product(fp16_dtypes, fp8_dtypes)
            ):
                dtype_out = dtype_q
                fname = f"batch_paged_decode_head_{head_dim}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}_idtype_{idtype}.cu"
                files_decode.append(str(path / fname))
                content = generate_batch_paged_decode_inst.get_cu_file_str(
                    head_dim,
                    pos_encoding_mode,
                    dtype_q,
                    dtype_kv,
                    dtype_out,
                    idtype,
                )
                for use_sliding_window in [True, False]:
                    for use_logits_soft_cap in [True, False]:
                        batch_decode_uris.append(
                            f"batch_decode_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_out}_"
                            f"dtype_idx_{idtype}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{use_sliding_window}_"
                            f"use_logits_cap_{use_logits_soft_cap}"
                        )
                write_if_different(path / fname, content)

    # single prefill files
    single_prefill_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
    ) in itertools.product(
        head_dims,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        mask_modes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            itertools.product(prefill_dtypes, fp8_dtypes)
        ):
            fname = f"single_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}.cu"
            files_prefill.append(str(path / fname))
            content = generate_single_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
            )
            for use_sliding_window in [True, False]:
                for use_logits_soft_cap in [True, False]:
                    if (
                        mask_mode == 0
                    ):  # NOTE(Zihao): uri do not contain mask, avoid duplicate uris
                        single_prefill_uris.append(
                            f"single_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{use_sliding_window}_"
                            f"use_logits_cap_{use_logits_soft_cap}_"
                            f"f16qk_{bool(allow_fp16_qk_reduction)}"
                        )
            write_if_different(path / fname, content)

    # batch prefill files
    batch_prefill_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
        idtype,
    ) in itertools.product(
        head_dims,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        mask_modes,
        idtypes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            itertools.product(prefill_dtypes, fp8_dtypes)
        ):
            fname = f"batch_paged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            files_prefill.append(str(path / fname))
            content = generate_batch_paged_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            fname = f"batch_ragged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            files_prefill.append(str(path / fname))
            content = generate_batch_ragged_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            for sliding_window in [True, False]:
                for logits_soft_cap in [True, False]:
                    if (
                        mask_mode == 0
                    ):  # NOTE(Zihao): uri do not contain mask, avoid duplicate uris
                        batch_prefill_uris.append(
                            f"batch_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"dtype_idx_{idtype}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{sliding_window}_"
                            f"use_logits_cap_{logits_soft_cap}_"
                            f"f16qk_{bool(allow_fp16_qk_reduction)}"
                        )

    # Change to relative path
    this_dir = pathlib.Path(__file__).parent.resolve()
    files_prefill = [str(pathlib.Path(p).relative_to(this_dir)) for p in files_prefill]
    files_decode = [str(pathlib.Path(p).relative_to(this_dir)) for p in files_decode]

    return (
        files_prefill,
        files_decode,
        single_decode_uris
        + batch_decode_uris
        + single_prefill_uris
        + batch_prefill_uris,
    )


def get_version():
    version = os.getenv("FLASHINFER_BUILD_VERSION")
    if version is None:
        with open(root / "version.txt") as f:
            version = f.read().strip()
    return version


def get_cuda_version() -> Tuple[int, int]:
    if not build_cuda_sources:
        return 0, 0

    if torch_cpp_ext.CUDA_HOME is None:
        nvcc = "nvcc"
    else:
        nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
    txt = subprocess.check_output([nvcc, "--version"], text=True)
    major, minor = map(int, re.findall(r"release (\d+)\.(\d+),", txt)[0])
    return major, minor


def generate_build_meta() -> None:
    d = {}
    version = get_version()
    d["cuda_major"], d["cuda_minor"] = get_cuda_version()
    d["torch"] = torch.__version__
    d["python"] = platform.python_version()
    d["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    with open(root / "python" / "flashinfer" / "_build_meta.py", "w") as f:
        f.write(f"__version__ = {version!r}\n")
        f.write(f"build_meta = {d!r}")


def generate_aot_config(aot_kernel_uris: List[str]) -> None:
    aot_config_str = f"""prebuilt_ops_uri = set({aot_kernel_uris})"""
    with open(root / "python" / "flashinfer" / "jit" / "aot_config.py", "w") as f:
        f.write(aot_config_str)


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


class NinjaBuildExtension(torch_cpp_ext.BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            max_num_jobs_cores = max(1, os.cpu_count())
            os.environ["MAX_JOBS"] = str(max_num_jobs_cores)

        super().__init__(*args, **kwargs)


@contextlib.contextmanager
def link_data_files() -> Iterator[None]:
    this_dir = pathlib.Path(__file__).parent
    data_dir = root / "python" / "flashinfer" / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    def ln(src: str, dst: str, is_dir: bool = False) -> None:
        (data_dir / dst).symlink_to(root / src, target_is_directory=is_dir)

    ln("3rdparty/cutlass", "cutlass", True)
    ln("include", "include", True)
    ln("python/csrc", "csrc", True)
    ln("version.txt", "version.txt")
    (this_dir / "MANIFEST.in").unlink(True)
    (this_dir / "MANIFEST.in").symlink_to("jit_MANIFEST.in")

    yield

    if sys.argv[1] != "develop":
        shutil.rmtree(data_dir)
        (this_dir / "MANIFEST.in").unlink(True)


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")


def get_macros_and_flags():
    define_macros = []
    extra_compile_args = {"cxx": [
        "-O3",
        "-Wno-switch-bool",
        "-Wno-unknown-pragmas",
        "-march=native",
        "-fopenmp",
        ]
    }
    extra_compile_args_sm90 = {}

    if build_cuda_sources:
        extra_compile_args["nvcc"] = [
            "-O3",
            "-std=c++17",
            "--threads=1",
            "-Xfatbin",
            "-compress-all",
            "-use_fast_math",
        ]

        extra_compile_args_sm90 = copy.deepcopy(extra_compile_args)
        extra_compile_args_sm90["nvcc"].extend(
            "-gencode arch=compute_90a,code=sm_90a".split()
        )

    return define_macros, extra_compile_args, extra_compile_args_sm90


if __name__ == "__main__":
    if build_cuda_sources:
        check_cuda_arch()

    remove_unwanted_pytorch_nvcc_flags()
    generate_build_meta()
    files_prefill, files_decode, aot_kernel_uris = get_instantiation_cu()
    generate_aot_config(aot_kernel_uris)

    include_dirs = []
    cuda_include_dirs = [
        str(root.resolve() / "include"),
        str(root.resolve() / "3rdparty" / "cutlass" / "include"),  # for group gemm
        str(root.resolve() / "3rdparty" / "cutlass" / "tools" / "util" / "include"),
    ]

    sources = [
        "csrc/cpu/activation.cpp",
        "csrc/cpu/norm.cpp",
        "csrc/cpu/decode.cpp",
        "csrc/cpu/gemm.cpp",
        "csrc/cpu/topk.cpp",
        "csrc/cpu/moe.cpp",
        "csrc/cpu/flashinfer_ops.cpp",
    ]
    cuda_sources = [
        "csrc/bmm_fp8.cu",
        "csrc/cascade.cu",
        "csrc/group_gemm.cu",
        "csrc/norm.cu",
        "csrc/page.cu",
        "csrc/quantization.cu",
        "csrc/rope.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
        "csrc/batch_decode.cu",
        "csrc/batch_prefill.cu",
        "csrc/single_decode.cu",
        "csrc/single_prefill.cu",
        "csrc/flashinfer_ops.cu",
    ]
    cuda_sources += files_decode
    cuda_sources += files_prefill

    define_macros, extra_compile_args, extra_compile_args_sm90 = get_macros_and_flags()
    # debug print
    print(define_macros, extra_compile_args, extra_compile_args_sm90)

    if build_cuda_sources:
        Extension = torch_cpp_ext.CUDAExtension
        sources += cuda_sources
        include_dirs += cuda_include_dirs
    else:
        Extension = torch_cpp_ext.CppExtension

    ext_modules = []
    ext_modules.append(
        Extension(
            name="flashinfer._kernels",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    )
    #ext_modules.append(
    #    torch_cpp_ext.CUDAExtension(
    #        name="flashinfer._kernels_sm90",
    #        sources=[
    #            "csrc/group_gemm_sm90.cu",
    #            "csrc/flashinfer_gemm_sm90_ops.cu",
    #        ],
    #        include_dirs=include_dirs,
    #        extra_compile_args=extra_compile_args_sm90,
    #    )
    #)

    # Suppress warnings complaining that:
    # Package 'flashinfer.data*' is absent from the `packages` configuration.
    warnings.filterwarnings("ignore", r".*flashinfer\.data.*", UserWarning)

    with link_data_files():
        setuptools.setup(
            name="flashinfer",
            version=get_version(),
            packages=setuptools.find_packages(
                include=["flashinfer*"],
                exclude=["flashinfer.data*"],
            ),
            include_package_data=True,
            author="FlashInfer team",
            license="Apache License 2.0",
            description="FlashInfer: Kernel Library for LLM Serving",
            url="https://github.com/flashinfer-ai/flashinfer",
            python_requires=">=3.8",
            ext_modules=ext_modules,
            cmdclass={"build_ext": NinjaBuildExtension},
        )
