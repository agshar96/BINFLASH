from fused_attention import attention as openai
from correct_fused_attention import attention_correct as openai_crrct
from torchMath_attention import torchMath_attention as torchMath
from customMask_attention import attention_masked as masked_attn
from blockmask_attention import attention_binaryBlkMat as blockmask_attn
import torch
import triton
import numpy as np

BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(8, 15)],
                line_arg="provider",
                line_vals=["openai", "openai_crrct", "torchMath", "masked_attn", "blck_masked"], #, "blck_masked_rcm"],
                line_names=["OpenAI Incorrect", "OpenAI Corrected", "TorchMath", "Masked Attention", "Block Masked"], #"Blck Masked RCM"],
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
        elif provider == "openai_crrct":
            fn = lambda: openai_crrct(q, k, v, causal, sm_scale)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "torchMath":
            fn = lambda: torchMath(q, k, v, sm_scale)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "masked_attn":
            file_name = "saved_mats_gen/mat_n_blks_" + str(N_CTX) + ".npz"
            maskMat = torch.tensor(np.load(file_name)['maskMat'], dtype=dtype, device=device)
            fn = lambda: masked_attn(q, k, v, causal, sm_scale, maskMat)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "blck_masked":
            file_name = "saved_mats_gen/mat_n_blks_" + str(N_CTX) + ".npz"
            data = np.load(file_name)
            maskMat_blk = torch.tensor(data['maskMat_blk'], dtype=dtype, device=device)
            maskMat = torch.tensor(data['maskMat'], dtype=dtype, device=device)
            maskMat_blk_T = torch.tensor(data['maskMat_blk_T'], dtype=dtype, device=device)
            # maskMat_blk = maskMat_blk.T
            # maskMat = maskMat.T
            fn = lambda: blockmask_attn(q, k, v, causal, sm_scale, maskMat, maskMat_blk, maskMat_blk_T)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "blck_masked_rcm":
            file_name = "saved_mats_gen/mat_n_blks_" + str(N_CTX) + ".npz"
            data = np.load(file_name)
            maskMat_RCM = torch.tensor(data['maskMat_RCM'], dtype=dtype, device=device)
            maskMat_RCM_blk = torch.tensor(data['maskMat_RCM_blk'], dtype=dtype, device=device)
            maskMat_RCM_blk_T = torch.tensor(data['maskMat_RCM_blk_T'], dtype=dtype, device=device)
            fn = lambda: blockmask_attn(q, k, v, causal, sm_scale, maskMat_RCM, maskMat_RCM_blk, maskMat_RCM_blk_T)
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
    bench_flash_attention.run(save_path="bench_result/binary_blk_run", print_data=True)