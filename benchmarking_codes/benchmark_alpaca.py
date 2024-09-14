'''
This script is used to benchmark the two masks for ALPACA dataset task.
Here we will use Dense Binary Block Mask, No Binary Block Mask, and Base Flash Attention.
'''
import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
#sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')

from alpaca_datasets.alpaca_full_and_output import return_full_alpaca_mask
from alpaca_datasets.prefix_llm_mask import return_prefix_llm_mask
from fused_attention import attention as openai
from binBlkMask_codes.dense_binBlkMask import attention_dense_binaryBlkMat as dense_binBlkMask
from customMask_attention import attention_masked as naiveAttnMsk
from triton_kernels.binBlkMask_kernels import return_binBlk_matrices
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
                line_vals=["openai","naiveAttnMsk" ,"binBlkMsk"], #, "blck_masked_rcm"],
                line_names=["Base Flash Attention", "Naive Attention Masking", "Binary Block Masking"],
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="ms",
                plot_name=f"Alpaca_Prefix-{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
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
        elif provider == "naiveAttnMsk":
            maskMat = return_prefix_llm_mask(N_CTX) #return_full_alpaca_mask(N_CTX)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            fn = lambda: naiveAttnMsk(q, k, v, causal, sm_scale, maskMat)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "binBlkMsk":
            maskMat = return_prefix_llm_mask(N_CTX) #return_prefix_llm_mask(N_CTX)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            maskMat_blk, num_ones, offset = return_binBlk_matrices(maskMat, is_dense=True)
            maskMat_blk_T, num_ones_T, offset_T = return_binBlk_matrices(maskMat.T, is_dense=True)

            fn = lambda: dense_binBlkMask(q, k, v, causal, sm_scale, maskMat, maskMat_blk, maskMat_blk_T,
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
    bench_flash_attention.run(save_path="benchresult_applications/alpaca_Prefix_A100", print_data=True)
