import torch
from utils import create_symmetric_sparse_matrix, plot_binary_matrix, create_stride_and_idx_mat
import numpy as np

matrix = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]])

inp_matrix = torch.tensor(matrix)
BLKSZE_I = 3
BLKSZE_J = 3
stride_matrix, Idx_matrix = create_stride_and_idx_mat(inp_matrix, BLKSZE_I, BLKSZE_J)
print(stride_matrix)
print(Idx_matrix)