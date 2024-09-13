'''
The purpose of this file is to test the functions in utils.py.
We will specifically test four functions:
1) get_causal_matrix
2) get_qk_causal_matrix
3) get_hash_causal_matrix
4) get_num_ones_and_offset
'''


import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_causal_matrix, plot_binary_matrix, get_qk_causal_matrix, get_hash_causal_matrix, get_num_ones_and_offset

# Test get_causal_matrix
# N = 1024
# causal_matrix = get_causal_matrix(N)
# plot_binary_matrix(causal_matrix)

# Test get_qk_causal_matrix
# N = 1024
# num_deletes = 512
# causal_matrix = get_qk_causal_matrix(N, num_deletes)
# print(causal_matrix.shape)
# plot_binary_matrix(causal_matrix)

# Test get_hash_causal_matrix
# N = 1024
# n_min = 8
# n_max = 256
# causal_matrix = get_hash_causal_matrix(N, n_min, n_max)
# print(causal_matrix.shape) 
# plot_binary_matrix(causal_matrix)

# Test get_num_ones_and_offset
N = 1024
n_min = 8
n_max = 256
causal_matrix = get_hash_causal_matrix(N, n_min, n_max)
plot_binary_matrix(causal_matrix)
num_ones, offset = get_num_ones_and_offset(causal_matrix, 128, 32)
print(f"num_ones: {num_ones},\n offset: {offset}")