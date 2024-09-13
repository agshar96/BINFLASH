import sys
sys.path.append('/home/agniv/Documents/BINFLASH/')

import timeit
from utils import *
import torch
import time

import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def calculate_sum_matrix(input_mat, output_sum_matrix, output_binblk_matrix,
                         input_stride_0, input_stride_1,
                         output_stride_0, output_stride_1,
                         BLKSIZE_I: tl.constexpr, BLKSIZE_J: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Get the relevant block from the input matrix
    start_m = row * BLKSIZE_I
    start_n = col * BLKSIZE_J

    offs_m = start_m + tl.arange(0, BLKSIZE_I)
    offs_n = start_n + tl.arange(0, BLKSIZE_J)

    input_ptrs = input_mat + offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1
    input_block = tl.load(input_ptrs)

    # Calculate the sum of the block
    sum = tl.sum(input_block)
    
    # print("row_offset", row_offset.shape[1])
    # Store the sum in the output sum matrix
    output_sum_ptr = output_sum_matrix + row * output_stride_0 + col * output_stride_1
    tl.store(output_sum_ptr, sum)

    # Store the binary block in the output binBlk matrix
    output_binblk_ptr = output_binblk_matrix + row * output_stride_0 + col * output_stride_1
    tl.store(output_binblk_ptr, sum > 0)

def return_sum_matrix(test_mat, BLKSIZE_I = 128, BLKSIZE_J = 32):

    sum_matrix = torch.zeros((test_mat.size(0)//BLKSIZE_I, test_mat.size(1)//BLKSIZE_J))
    binblk_matrix = torch.zeros_like(sum_matrix)
    grid = (sum_matrix.size(0), sum_matrix.size(1))
    sum_matrix = sum_matrix.to('cuda')
    test_mat = test_mat.to('cuda')
    binblk_matrix = binblk_matrix.to('cuda')
    calculate_sum_matrix[grid](test_mat, sum_matrix, binblk_matrix,
                               test_mat.stride(0), test_mat.stride(1),
                               sum_matrix.stride(0), sum_matrix.stride(1),
                               BLKSIZE_I, BLKSIZE_J)
    return sum_matrix, binblk_matrix

def return_sum_matrix_v2(testMat,BLKSIZE_I = 128, BLKSIZE_J = 32):
    # Get the dimensions of the input matrix
    nrows, ncols = testMat.shape
    
    # Reshape the matrix to create blocks
    reshaped = testMat.view(nrows // BLKSIZE_I, BLKSIZE_I, -1, BLKSIZE_J)
    
    # Sum the elements in each block
    block_sums = reshaped.sum(dim=(1, 3))
    
    return block_sums

@triton.jit
def calculate_ones_and_offset(input_sum_matrix, output_ones_matrix, output_offset_matrix,
                              input_stride_0, input_stride_1,
                              output_stride_0, output_stride_1,
                              SUM_CHECK: tl.constexpr,
                              COL_SIZE: tl.constexpr):
    row = tl.program_id(0)

    # Get the relevant block from the input matrix
    offs_n = tl.arange(0, COL_SIZE)
    input_ptrs = input_sum_matrix + row * input_stride_0 + offs_n[None, :] * input_stride_1
    input_block = tl.load(input_ptrs)

    # Check complete one block
    cond_inp_blk = tl.where(input_block == SUM_CHECK, 1, 0)

    # Store total complete ones in the output ones matrix
    tot_ones = tl.sum(cond_inp_blk)
    output_ones_ptr = output_ones_matrix + row * output_stride_0
    tl.store(output_ones_ptr, tot_ones)

    # Store the offset in the output offset matrix
    first_one = tl.argmax(cond_inp_blk, axis=1)
    off_ptr = tl.arange(0, 1)
    output_offset_ptr = output_offset_matrix + row * output_stride_0 + off_ptr * output_stride_1
    tl.store(output_offset_ptr, first_one)

def return_ones_and_offset(sum_matrix, SUM_CHECK):
    ones_mat = torch.zeros((sum_matrix.size(0), 1))
    offset_mat = torch.zeros_like(ones_mat)
    grid = (sum_matrix.size(0), 1)
    ones_mat = ones_mat.to('cuda')
    offset_mat = offset_mat.to('cuda')
    COL_SIZE = sum_matrix.size(1)
    calculate_ones_and_offset[grid](sum_matrix, ones_mat, offset_mat,
                                    sum_matrix.stride(0), sum_matrix.stride(1),
                                    ones_mat.stride(0), ones_mat.stride(1),
                                    SUM_CHECK, COL_SIZE)
    return ones_mat[:,0], offset_mat[:,0]

def return_binBlk_matrices(test_mat, BLKSIZE_I = 128, BLKSIZE_J = 32, is_dense = False):
    SUM_CHECK = BLKSIZE_I * BLKSIZE_J

    sum_matrix, binBlk_matrix = return_sum_matrix(test_mat, BLKSIZE_I, BLKSIZE_J)
    if is_dense:
        ones_matrix, offset_matrix = return_ones_and_offset(sum_matrix, SUM_CHECK)
        return binBlk_matrix, ones_matrix, offset_matrix
    return binBlk_matrix

test_mat = torch.ones(16384, 16384).to(torch.float16).to('cuda')

BLKSIZE_I = 128
BLKSIZE_J = 32
SUM_CHECK = BLKSIZE_I * BLKSIZE_J

# def timed_function():
#     torch.cuda.synchronize()  # Ensure previous GPU operations are complete
#     return_binBlk_matrices(test_mat)
#     torch.cuda.synchronize()  # Ensure all GPU operations are complete

# execution_time = timeit.timeit(timed_function, number=100)
# print(f"Average execution time over 100 runs: {execution_time / 100} seconds")

# start_time = time.time()

# # sum_matrix, binBlk_matrix = return_sum_matrix(test_mat, BLKSIZE_I, BLKSIZE_J)
# # print(sum_matrix.size(), binBlk_matrix.size())
# # ones_matrix, offset_matrix = return_ones_and_offset(sum_matrix, SUM_CHECK)

# # sum_matrix = return_sum_matrix_v2(test_mat, BLKSIZE_I, BLKSIZE_J)
# # print(sum_matrix.size())

# print(f"Time taken: {time.time() - start_time}")





