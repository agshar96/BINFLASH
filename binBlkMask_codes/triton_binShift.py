import torch

import triton
import triton.language as tl

@triton.jit
def get_firstbits(data):
    count = 0x80000000
    firstbit = data & count
    MASK = True
    if firstbit:
        firstbit = tl.cast(1, tl.uint32)
    return firstbit

@triton.jit
def print_firstbits(data):
    pid = tl.program_id(0)
    for i in range(32):
        firstbit = get_firstbits(data)
        data = data << 1
        if firstbit == 1:
            j = 31 - i
            print("FIRSTBIT: ", j)

    # print("LOCAL: ", local_var)

data = -1073741824
output_var = 0
grid = (1,)
print_firstbits[grid](data)