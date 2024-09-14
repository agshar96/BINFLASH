import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
#sys.path.append('/home/agniv/Documents/BINFLASH/')

sys.path.append('..')
from triton_kernels.binBlkMask_kernels import return_binBlk_matrices, return_sum_matrix, return_sum_matrix_v2
from triton_kernels.binBlkMask_kernels_dontUse import return_binBlkMask

import torch
import triton
import numpy as np


ATCH, N_HEADS, HEAD_DIM = 4, 32, 64 # The batch size for us is number of sequences we are processing at one go thus this batch size is 1
configs = []

configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(8,15)],
        line_arg="provider",
        line_vals=["binBlk","Dense_binBlk"], #, "blck_masked_rcm"],
        line_names=["BinBlkMsk", "BinBlkMsk with total_ones and offset"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"kernel_bench",
        args={
                },
    ))

@triton.testing.perf_report(configs)
def bench_kernels(N_CTX, provider, device="cuda"):
    assert provider in ["binBlk","Dense_binBlk"]
    warmup = 250
    rep = 1000
    dtype = torch.float16

    try:
        if provider == "binBlk":
            testMat = torch.ones((N_CTX, N_CTX), dtype=dtype, device=device)
            # fn = lambda: return_sum_matrix(testMat)
            fn = lambda: return_binBlk_matrices(testMat, 128, 32)
            # o = fn
        elif provider == "Dense_binBlk":
            testMat = torch.ones((N_CTX, N_CTX), dtype=dtype, device=device)
            fn = lambda: return_binBlk_matrices(testMat, 128, 32, is_dense=True)
            # o = fn()
        else:
            raise ValueError("Invalid provider")
        print(f"Benchmarking (N={N_CTX} for {provider} ...")
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        # print("Time is: ", ms)
    except Exception as e:
        print(e)
        return
    return ms

if __name__ == "__main__":
    bench_kernels.run(save_path="benchresult_A100/kernels_A100/", print_data=True)
