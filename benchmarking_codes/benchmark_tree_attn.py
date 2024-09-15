'''
This script is used to benchmark the two masks for MEDUSA tree attention task.
Here we will use Dense Binary Block Mask, No Binary Block Mask, and Base Flash Attention.
We do it for two cases: one for total number of predictions per head and the other for multiple medusa batches
'''
import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
# sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')

from tree_attention.base_tree_attention import patched_tree_mask, get_nctx_from_batchSize, get_nctx_from_predictions, padded_tree_mask
from fused_attention import attention as openai
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
                x_names=["MEDUSA_HEADS"], #, "BATCH_SIZE"],MEDUSA_PREDICTIONS, MEDUSA_HEADS
                x_vals=[i for i in range(4, 9)], #, [2**i for i in range(0, 6)], i for i in range(3, 11), i for i in range(4, 9), i for i in range(3, 11)
                line_arg="provider",
                line_vals=["openai","naiveAttnMsk" ,"binBlkMsk"], #, "blck_masked_rcm"],
                line_names=["Base Flash Attention", "Naive Attention Masking", "Binary Block Masking"],
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="ms",
                plot_name=f"MEDUSA_HEADS-{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, MEDUSA_HEADS, HEAD_DIM, causal, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    sm_scale = 0.5

    # MEDUSA_HEADS = 4
    MEDUSA_PREDICTIONS = 3
    ## Getting N_CTX from batch size
    N_CTX = get_nctx_from_predictions(MEDUSA_HEADS, MEDUSA_PREDICTIONS) #
    # N_CTX = get_nctx_from_batchSize(BATCH_SIZE, MEDUSA_HEADS, MEDUSA_PREDICTIONS)
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
            # maskMat = patched_tree_mask(MEDUSA_HEADS, MEDUSA_PREDICTIONS, N_CTX)
            maskMat = padded_tree_mask(MEDUSA_HEADS, MEDUSA_PREDICTIONS)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            fn = lambda: naiveAttnMsk(q, k, v, causal, sm_scale, maskMat)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "binBlkMsk":
            # maskMat = patched_tree_mask(MEDUSA_HEADS, MEDUSA_PREDICTIONS, N_CTX)
            maskMat = padded_tree_mask(MEDUSA_HEADS, MEDUSA_PREDICTIONS)
            maskMat = torch.tensor(maskMat, dtype=dtype, device=device)
            maskMat_blk, num_ones, offset = return_binBlk_matrices(maskMat, is_dense=True)
            maskMat_blk_T, num_ones_T, offset_T = return_binBlk_matrices(maskMat.T, is_dense=True)

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
    bench_flash_attention.run(save_path="benchresult_applications/MEDUSA_HEADS/", print_data=True)