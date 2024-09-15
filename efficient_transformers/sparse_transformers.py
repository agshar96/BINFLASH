import numpy as np
import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
# sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')
from utils import plot_binary_matrix

def get_lowerWindow_mask(N, stride):
    mask = np.zeros((N, N))

    # Generate indices for rows and columns based on the stride
    row_indices = np.arange(N).reshape(-1, 1)
    col_indices = row_indices - np.arange(0, stride)

    row_indices = np.repeat(row_indices, col_indices.shape[1], axis=1)

    # Filter out-of-bounds indices
    valid_indices = col_indices >= 0

    # Set the appropriate positions in the mask to 1
    mask[row_indices[valid_indices], col_indices[valid_indices]] = 1
    return mask

def get_sparse_strided(N, stride = 8):
    # Initialize a mask of zeros
    mask = get_lowerWindow_mask(N, stride)

    # Generate indices for rows and columns based on the stride
    row_indices = np.arange(N).reshape(-1, 1)
    col_indices = row_indices - np.arange(0, N, stride)

    row_indices = np.repeat(row_indices, col_indices.shape[1], axis=1)

    # Filter out-of-bounds indices
    valid_indices = col_indices >= 0

    # Set the appropriate positions in the mask to 1
    mask[row_indices[valid_indices], col_indices[valid_indices]] = 1

    return mask
    

def get_sparse_fixed(N, stride = 8, c = 3):
    
    triangle_mask = np.tril(np.ones((stride, stride)))
    mask = np.zeros((N, N))
    for i in range(0, N, stride):
        mask[i:i+stride, i:i+stride] = triangle_mask
        cur_col = i + stride

        mask[:, cur_col-c:cur_col] = 1
    
    return np.tril(mask)