import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
sys.path.append('/home/agniv/Documents/BINFLASH/')

import numpy as np
import torch
from triton_kernels.binBlkMask_kernels import return_binBlk_matrices
from tree_attention.base_tree_attention import create_tree_mask
from efficient_transformers.longformer import get_global_mask
from utils import *
from PIL import Image

def plot_grid_and_save(matrix, BLKSIZE_I, BLKSIZE_J, path):
    N = matrix.shape[0]
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap='binary', interpolation='none')
    plt.xticks(np.arange(-0.5, N, BLKSIZE_J), visible=False)
    plt.yticks(np.arange(-0.5, N, BLKSIZE_I), visible=False)
    plt.grid(color='red', linewidth=1, alpha=0.75)
    plt.xlim(0, N-0.5)
    plt.ylim(N-0.5, 0)
    plt.tick_params(bottom=False, left=False)
    # plt.tick_params(visible=True)
    plt.savefig(path)
    plt.close()

def plot_and_save_binary_matrix(matrix, path, N = None, M = None, enable_grid = True):
    N = matrix.shape[0] if N is None else N
    M = matrix.shape[1] if M is None else M

    matrix = matrix > 0
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap='binary', interpolation='None')
    plt.xticks(np.arange(-0.5, M, 1), visible=False)
    plt.yticks(np.arange(-0.5, N, 1), visible=False)
    if enable_grid == True:
        

        plt.grid(color='black', linewidth=0.5, alpha=0.5)
    plt.tick_params(bottom=False, left=False)

    plt.xlim(-0.5, M - 0.5)
    plt.ylim(N -0.5, -0.5)

    plt.savefig(path)
    # plt.show()
    plt.close()

path = 'saved_figures/LongFormer/'

matrix = get_global_mask(32, 3)

plot_and_save_binary_matrix(matrix, path + 'Global_fixed.png', enable_grid = False)
# plot_grid_and_save(hash_mat.cpu().numpy(), 16, 16, path + 'base_matrix_blocked.png')

# binBlk_matrix = return_binBlk_matrices(hash_mat, 16, 16)

# plot_and_save_binary_matrix(binBlk_matrix.cpu().numpy(), path + 'binaryBlkMatrix.png', enable_grid = True)

