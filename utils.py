import torch
import numpy as np
import matplotlib.pyplot as plt
import math

def create_symmetric_sparse_matrix(N = 1024, base_prob=0.5, decay=0.1, pow = 10):
    """
    Create an NxN symmetric binary sparse matrix where the farther from the diagonal, the lesser the probability of having 1.
    
    Parameters:
    N (int): Size of the matrix.
    base_prob (float): Base probability for the diagonal elements.
    decay (float): Decay rate for the probability as we move away from the diagonal.
    pow (int): Log2 of N
    
    Returns:
    np.ndarray: Symmetric binary sparse matrix of size NxN.
    """
    # Initialize the matrix with zeros
    matrix = np.zeros((N, N), dtype=int)
    
    # Iterate over the upper triangular part including the diagonal
    for i in range(N):
        for j in range(i, N):
            if i == j:
                matrix[i, j] = 1
                continue
            # Calculate the probability based on the distance from the diagonal
            distance = abs(i - j)

            prob = max(base_prob * np.exp(-decay * distance), 0.0002)
            # prob = 8 / (1000 * round(math.log2(distance + 1)) * (pow * 0.5))
            
            # Fill the matrix with 1 based on the probability
            if np.random.rand() < prob:
                matrix[i, j] = 1
                if i != j:
                    matrix[j, i] = 1  # Ensure symmetry
    
    return matrix

def plot_binary_matrix(matrix):
    '''
    Function to plot an NxN binary matrix as black and white values
    '''
    N = matrix.shape[0]
    plt.imshow(matrix, cmap='binary', interpolation='none')
    # plt.xticks(np.arange(0, N, 1))
    # plt.yticks(np.arange(0, N, 1))
    # plt.grid(color='black', linewidth=0.5)
    plt.show()

def big_matrix_from_diag(small_mat, N):
    '''
    Function takes a n x n dimension matrix and adds it along the diagonal of a bigger matrix
    Args:
        small_mat: n x n matrix
        N: size of the bigger matrix (N %n == 0)
    '''
    assert N % small_mat.shape[0] == 0, "N should be a multiple of the small matrix size"
    n = small_mat.shape[0]
    big_mat = torch.zeros(N, N)
    for i in range(N//n):
        big_mat[i*n:(i+1)*n, i*n:(i+1)*n] = small_mat
    return big_mat

def get_prefix_sum_matrix(matrix):
    '''
        Build the prefix sum matrix (pre) where the sum up to each point (i, j) is calculated 
        by adding the current element to the sum of the elements above and to the left
    '''
    pre = torch.zeros_like(matrix)  # prefix sum matrix
    pre[0, 0] = matrix[0, 0]
    for i in range(1, matrix.shape[1]):
        pre[0, i] = pre[0, i-1] + matrix[0, i]
    for i in range(1, matrix.shape[0]):
        pre[i, 0] = pre[i-1, 0] + matrix[i, 0]
        for j in range(1, matrix.shape[1]):
            pre[i, j] = pre[i-1, j] + pre[i, j-1] - pre[i-1, j-1] + matrix[i, j]
    return pre

def get_sum_from_prefix_sum_matrix(pre, i, j, n_i, n_j):
    '''
        Get the sum of the elements in the submatrix of size n_i x n_j with bottom right corner at (i, j)
    '''
    if i - n_i < 0:
        if j - n_j < 0:
            return pre[i, j]
        else:
            return pre[i, j] - pre[i, j-n_j]
    elif j - n_j < 0:
        if i - n_i < 0:
            return pre[i, j]
        else:
            return pre[i, j] - pre[i-n_i, j]
    else:
        # print("Inside i, j", i, j)
        # print("Inside i-n_i, j-n_j", i-n_i, j-n_j)
        return pre[i, j] - pre[i-n_i, j] - pre[i, j-n_j] + pre[i-n_i, j-n_j]

def create_blksum_matrix(inp_matrix, BLKSZE_I, BLKSZE_J):
    '''
        The function takes a binary matrix, calculates its prefix sum.
        Then it creates a matrix output_matrix of size (inp_matrix.shape[0]//BLKSZE_I, inp_matrix.shape[1]//BLKSZE_J)
        where each element is the sum of the elements in the submatrix of size BLKSZE_I x BLKSZE_J.
        Args:
            inp_matrix: binary matrix
            BLKSZE_I: size of the block in the row direction
            BLKSZE_J: size of the block in the column direction
    '''
    assert inp_matrix.shape[0] % BLKSZE_I == 0, "The number of rows should be a multiple of BLKSZE_I"
    assert inp_matrix.shape[1] % BLKSZE_J == 0, "The number of columns should be a multiple of BLKSZE_J"
    pre = get_prefix_sum_matrix(inp_matrix)
    print("Prefix sum matrix calculated")
    output_matrix = torch.zeros(inp_matrix.shape[0]//BLKSZE_I, inp_matrix.shape[1]//BLKSZE_J)
    for i in range(0, inp_matrix.shape[0], BLKSZE_I):
        for j in range(0, inp_matrix.shape[1], BLKSZE_J):
            # print("Calling for i, j", i, j)
            output_matrix[i//BLKSZE_I, j//BLKSZE_J] = get_sum_from_prefix_sum_matrix(pre, i+BLKSZE_I-1, j+BLKSZE_J-1, BLKSZE_I, BLKSZE_J)
        print("Finished row", i//BLKSZE_I)
    return output_matrix

def contains_one(matrix):
    for row in matrix:
        for element in row:
            if element == 1:
                return True
    return False

def create_blk_one_matrix(inp_matrix, BLKSZE_I, BLKSZE_J):
    '''
        The function takes a binary matrix and creates a matrix output_matrix of size (inp_matrix.shape[0]//BLKSZE_I, inp_matrix.shape[1]//BLKSZE_J)
        where each element is 1 if the submatrix of size BLKSZE_I x BLKSZE_J contains a 1, otherwise 0.
        Args:
            inp_matrix: binary matrix
            BLKSZE_I: size of the block in the row direction
            BLKSZE_J: size of the block in the column direction
    '''
    assert inp_matrix.shape[0] % BLKSZE_I == 0, "The number of rows should be a multiple of BLKSZE_I"
    assert inp_matrix.shape[1] % BLKSZE_J == 0, "The number of columns should be a multiple of BLKSZE_J"
    output_matrix = torch.zeros(inp_matrix.shape[0]//BLKSZE_I, inp_matrix.shape[1]//BLKSZE_J)
    for i in range(0, inp_matrix.shape[0], BLKSZE_I):
        for j in range(0, inp_matrix.shape[1], BLKSZE_J):
            # print("j is: ", j)
            output_matrix[i//BLKSZE_I, j//BLKSZE_J] = int(contains_one(inp_matrix[i:i+BLKSZE_I, j:j+BLKSZE_J]))
        print("Finished row", i//BLKSZE_I)
    return output_matrix

def return_non_zero_columns(matrix, offset):
    '''
        Returns the indices of the columns that contain a 1 in the matrix
    '''
    non_zero_columns = []
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            if matrix[i, j] == 1:
                non_zero_columns.append(offset + j)
                break
    return non_zero_columns

def pad_with_neg1(lst, n):
    '''
        Pads the list with zeros to make its length n
    '''
    return lst + [-1]*(n - len(lst))

def create_stride_and_idx_mat(inp_matrix, BLKSZE_I, BLKSZE_J):
    '''
        This function creates two matrices:
        1. stride_matrix: For each block of size BLKSZE_I x BLKSZE_J it stores number of non zero columns.
                          This can then be used to calculate stride for keyIdx_matrix.
                          Size of matrix: (inp_matrix.shape[0]//BLKSZE_I, inp_matrix.shape[1]//BLKSZE_J)
        2. idx_matrix: This matrix contains index of non-zero columns in BLKSZE_I number of rows. 
                            The columns are padded with -1 to be of length inp_matrix.shape[1]
    '''
    stride_matrix = torch.zeros(inp_matrix.shape[0]//BLKSZE_I, inp_matrix.shape[1]//BLKSZE_J)
    idx_matrix = torch.zeros(inp_matrix.shape[0]//BLKSZE_I, inp_matrix.shape[1])
    for i in range(0, inp_matrix.shape[0], BLKSZE_I):
        non_zero_columns = []
        for j in range(0, inp_matrix.shape[1], BLKSZE_J):
            columns = return_non_zero_columns(inp_matrix[i:i+BLKSZE_I, j:j+BLKSZE_J], j)
            stride_matrix[i//BLKSZE_I, j//BLKSZE_J] = len(columns)
            non_zero_columns.extend(columns)
        non_zero_columns = pad_with_neg1(non_zero_columns, inp_matrix.shape[1])
        idx_matrix[i//BLKSZE_I] = torch.tensor(non_zero_columns)
        print("Finished row", i//BLKSZE_I)
    return stride_matrix, idx_matrix


   


    