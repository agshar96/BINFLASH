import sys
sys.path.append('/home/agniv/Documents/BINFLASH/')

from utils import *
import torch
import time

import triton
import triton.language as tl
from triton_kernels.binBlkMask_kernels import return_sum_matrix, return_ones_and_offset

@triton.jit
def calculate_binBlkMask(inpMat, binBlkMat,
                        inpMat_stride_0, inpMat_stride_1,
                        binBlkMat_stride_0, binBlkMat_stride_1,
                        BLKSIZE_I: tl.constexpr, BLKSIZE_J: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    # Get output cell location
    out_row = row // BLKSIZE_I
    out_col = col // BLKSIZE_J

    # Get relevant mask value from input matrix
    mask_loc = inpMat + row * inpMat_stride_0 + col * inpMat_stride_1
    mask_val = tl.load(mask_loc)

    if mask_val:
        output_loc = binBlkMat + out_row * binBlkMat_stride_0 + out_col * binBlkMat_stride_1
        tl.store(output_loc, 1)



def return_binBlkMask(inpMat, BLKSIZE_I = 128, BLKSIZE_J = 32):
    binBlkMat = torch.zeros((inpMat.size(0)//BLKSIZE_I, inpMat.size(1)//BLKSIZE_J))
    grid = (inpMat.size(0), inpMat.size(1))
    inpMat = inpMat.to('cuda') 
    binBlkMat = binBlkMat.to('cuda')
    calculate_binBlkMask[grid](inpMat, binBlkMat,
                                inpMat.stride(0), inpMat.stride(1),
                                binBlkMat.stride(0), binBlkMat.stride(1),
                                BLKSIZE_I, BLKSIZE_J)
    return binBlkMat

# test_mat = create_symmetric_sparse_matrix(1024)
# BLKSIZE_I = 128
# BLKSIZE_J = 32

# # print(test_mat)

# test_mat = torch.from_numpy(test_mat).float()
# binBlkMat = return_binBlkMask(test_mat, BLKSIZE_I, BLKSIZE_J)
# sum_matrix, binBlk_matrix = return_sum_matrix(test_mat, BLKSIZE_I, BLKSIZE_J)

# print("Are they close: ", torch.allclose(binBlkMat, binBlk_matrix, atol=1e-2, rtol=0))

# test_mat = create_symmetric_sparse_matrix(1024)
# test_mat = torch.from_numpy(test_mat).float()

# BLKSIZE_I = 128
# BLKSIZE_J = 32
# SUM_CHECK = BLKSIZE_I * BLKSIZE_J

# start_time = time.time()

# sum_matrix, binBlk_matrix = return_sum_matrix(test_mat, BLKSIZE_I, BLKSIZE_J)
# ones_matrix, offset_matrix = return_ones_and_offset(sum_matrix, SUM_CHECK)

# print(f"Time taken: {time.time() - start_time}")