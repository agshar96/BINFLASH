import numpy as np
import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
# sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')
from utils import plot_binary_matrix

def get_windowed_mask(N, window_size):
    one_side_win = window_size // 2

    mask = np.zeros((N, N))

    # Generate indices for rows and columns based on the stride
    row_indices = np.arange(N).reshape(-1, 1)
    col_indices = row_indices - np.arange(0, one_side_win+1)

    row_indices = np.repeat(row_indices, col_indices.shape[1], axis=1)

    # Filter out-of-bounds indices
    valid_indices = col_indices >= 0

    # Set the appropriate positions in the mask to 1
    mask[row_indices[valid_indices], col_indices[valid_indices]] = 1

    mask = np.logical_or(mask, mask.T)

    return mask

def get_dilated_window(N, window_size, dilation=1):
    one_side_win = window_size // 2

    mask = np.zeros((N, N))

    # Generate indices for rows and columns based on the stride
    row_indices = np.arange(N).reshape(-1, 1)
    col_indices = row_indices - np.arange(0, dilation*one_side_win + 1, dilation)

    row_indices = np.repeat(row_indices, col_indices.shape[1], axis=1)

    # Filter out-of-bounds indices
    valid_indices = col_indices >= 0

    # Set the appropriate positions in the mask to 1
    mask[row_indices[valid_indices], col_indices[valid_indices]] = 1

    mask = np.logical_or(mask, mask.T)

    return mask

def get_global_mask(N, window_size):

    num_of_CLS = int(np.log2(N)) - 2

    mask = get_windowed_mask(N, window_size)

    mask[:, :2] = 1
    mask[:2, :] = 1

    indices = np.arange(2, N)
    permuted_arr = np.random.permutation(indices)[:num_of_CLS]
    for i in permuted_arr:
        mask[i, :] = 1
        mask[:, i] = 1

    return mask