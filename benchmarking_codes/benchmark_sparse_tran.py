import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
# sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')

from efficient_transformers.sparse_transformers import get_sparse_strided, get_sparse_fixed
from correct_fused_attention import attention_correct as openai
# from blockmask_attention import attention_binaryBlkMat as binBlkMsk_attn
from binBlkMask_codes.base_binBlkMask import attention_binaryBlkMat as binBlkMsk_attn
from customMask_attention import attention_masked as naiveAttnMsk
from triton_kernels.binBlkMask_kernels import return_binBlk_matrices
import torch
import triton
import numpy as np

BATCH, N_HEADS, HEAD_DIM = 4, 32, 64 # The batch size for us is number of sequences we are processing at one go thus this batch size is 1
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(8, 15)],
                line_arg="provider",
                line_vals=["openai_s","naiveAttnMsk_s" ,"binBlkMsk_s", "openai_f","naiveAttnMsk_f" ,"binBlkMsk_f"], #, "blck_masked_rcm"],
                line_names=["OpenAI FlashAttn(Strided)", "Naive Attention Mask(Strided)", "Binary Block Mask(Strided)",
                            "OpenAI FlashAttn(Fixed)", "Naive Attention Mask(Fixed)", "Binary Block Mask(Fixed)"],
                styles=[("red", "-."), ("blue", "-."), ("green", "-."),
                        ("red", ":"), ("blue", ":"), ("green", ":")],
                ylabel="ms",
                plot_name=f"Sparse_Transformer-{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
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
        if provider == "openai_s":
            # causal = False
            fn = lambda: openai(q, k, v, causal, sm_scale)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "naiveAttnMsk_s":
            maskMat = get_sparse_strided(N_CTX)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            fn = lambda: naiveAttnMsk(q, k, v, causal, sm_scale, maskMat)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "binBlkMsk_s":
            maskMat =  get_sparse_strided(N_CTX)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            maskMat_blk = return_binBlk_matrices(maskMat)
            maskMat_blk_T = return_binBlk_matrices(maskMat.T)

            fn = lambda: binBlkMsk_attn(q, k, v, causal, sm_scale, maskMat, maskMat_blk, maskMat_blk_T)

            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "openai_f":
            fn = lambda: openai(q, k, v, causal, sm_scale)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "naiveAttnMsk_f":
            maskMat = get_sparse_fixed(N_CTX)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            fn = lambda: naiveAttnMsk(q, k, v, causal, sm_scale, maskMat)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "binBlkMsk_f":
            maskMat = get_sparse_fixed(N_CTX)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            maskMat_blk = return_binBlk_matrices(maskMat)
            maskMat_blk_T = return_binBlk_matrices(maskMat.T)

            fn = lambda: binBlkMsk_attn(q, k, v, causal, sm_scale, maskMat, maskMat_blk, maskMat_blk_T)

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
    bench_flash_attention.run(save_path="benchresult_applications/Sparse_Transformer_v2/", print_data=True)
