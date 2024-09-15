'''
This script is used to benchmark the binary block masking algorithms against base flash attention.
The masks are selected as causal, therefore only base OpenAI model is used, not corrected one.
'''
import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
# sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')

from fused_attention import attention as openai
from binBlkMask_codes.base_binBlkMask import attention_binaryBlkMat as base_binBlkMask
from binBlkMask_codes.transition_binBlkMask import attention_transition_binaryBlkMask as transition_binBlkMask
from binBlkMask_codes.dense_binBlkMask import attention_dense_binaryBlkMat as dense_binBlkMask
from binBlkMask_codes.bitShift_binBlkMask import attention_bitShift_binBlkMask as bitShift_binBlkMask
from binBlkMask_codes.combined_binBlkMask import attention_combined_binBlkMask as combined_binBlkMask
from binBlkMask_codes.combine_wo_binShift import attention_combined_wo_binBlkMask as combined_wo_binBlkMask

import torch
import triton
import numpy as np


BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(8, 15)],
                line_arg="provider",
                line_vals=["openai", "base_binBlkMask", "transition_binBlkMask", "dense_binBlkMask", "bitShift_binBlkMask", "combined_binBlkMask"], #, "blck_masked_rcm"],
                line_names=["OpenAI FlashAttn", "Base Binary Mask", "Transition Check Binary Mask", "On-Band Binary Mask", 
                            "BitShift Binary Mask", "Combined Binary Mask"], #"Blck Masked RCM"],
                styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-"), ("purple", "-"), ("brown", "-")],
                ylabel="ms",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    sm_scale = 0.5
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    try:
        if provider == "openai":
            causal = True
            fn = lambda: openai(q, k, v, causal, sm_scale)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "base_binBlkMask":
            file_name = "saved_mats/causal/mat_n_blks_" + str(N_CTX) + ".pt"
            data = torch.load(file_name)
            maskMat = data['maskMat'].to(dtype=dtype).to("cuda")
            maskMat_blk = data['maskMat_blk'].to(dtype=dtype).to("cuda")
            maskMat_blk_T = data['maskMat_blk_T'].to(dtype=dtype).to("cuda")
            fn = lambda: base_binBlkMask(q, k, v, causal, sm_scale, maskMat, maskMat_blk, maskMat_blk_T)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "transition_binBlkMask":
            file_name = "saved_mats/causal/mat_n_blks_" + str(N_CTX) + ".pt"
            data = torch.load(file_name)
            maskMat = data['maskMat'].to(dtype=dtype).to("cuda")
            maskMat_blk = data['maskMat_blk'].to(dtype=dtype).to("cuda")
            maskMat_blk_T = data['maskMat_blk_T'].to(dtype=dtype).to("cuda")
            fn = lambda: transition_binBlkMask(q, k, v, causal, sm_scale, maskMat, maskMat_blk, maskMat_blk_T)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "dense_binBlkMask":
            file_name = "saved_mats/causal/mat_n_blks_" + str(N_CTX) + ".pt"
            data = torch.load(file_name)
            maskMat = data['maskMat'].to(dtype=dtype).to("cuda")
            maskMat_blk = data['maskMat_blk'].to(dtype=dtype).to("cuda")
            maskMat_blk_T = data['maskMat_blk_T'].to(dtype=dtype).to("cuda")
            num_ones = data['num_ones'].to(dtype=dtype).to("cuda")
            offset = data['offset'].to(dtype=dtype).to("cuda")
            num_ones_T = data['num_ones_T'].to(dtype=dtype).to("cuda")
            offset_T = data['offset_T'].to(dtype=dtype).to("cuda")

            fn = lambda: dense_binBlkMask(q, k, v, causal, sm_scale, maskMat, maskMat_blk, maskMat_blk_T,
                                          num_ones, offset, num_ones_T, offset_T)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "bitShift_binBlkMask":
            file_name = "saved_mats/causal/mat_n_blks_" + str(N_CTX) + ".pt"
            data = torch.load(file_name)
            maskMat = data['maskMat'].to(dtype=dtype).to("cuda")
            maskMat_int16 = data['maskMat_int16'].to(dtype=torch.int16).to("cuda")
            maskMat_int16_T = data['maskMat_int16_T'].to(dtype=torch.int16).to("cuda")
            fn = lambda: bitShift_binBlkMask(q, k, v, causal, sm_scale, maskMat,
                                          maskMat_int16, maskMat_int16_T)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "combined_binBlkMask":
            file_name = "saved_mats/causal/mat_n_blks_" + str(N_CTX) + ".pt"
            data = torch.load(file_name)
            maskMat = data['maskMat'].to(dtype=dtype).to("cuda")
            maskMat_blk = data['maskMat_blk'].to(dtype=dtype).to("cuda")
            maskMat_blk_T = data['maskMat_blk_T'].to(dtype=dtype).to("cuda")
            num_ones = data['num_ones'].to(dtype=dtype).to("cuda")
            offset = data['offset'].to(dtype=dtype).to("cuda")
            num_ones_T = data['num_ones_T'].to(dtype=dtype).to("cuda")
            offset_T = data['offset_T'].to(dtype=dtype).to("cuda")
            fn = lambda: combined_wo_binBlkMask(q, k, v, causal, sm_scale, maskMat,
                                          maskMat_blk, maskMat_blk_T,
                                          num_ones, offset, num_ones_T, offset_T)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)

        print(f"Benchmarking {mode} (N={N_CTX}, H={H}, B={BATCH}, d={HEAD_DIM}) for {provider} ...")
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    except torch.cuda.OutOfMemoryError:
            ms = float('NaN')
    except triton.runtime.errors.OutOfResources:
        ms = float('NaN')
    except RuntimeError as e:
        print(f"{provider} raised the following error:")
        print(e)
        ms = float('NaN')
    return ms

if __name__ == "__main__":
    bench_flash_attention.run(save_path="bench_result_newChanges/casual_retest/", print_data=True)
