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

            prob = max(base_prob * np.exp(-decay * distance), 0.0002) # Illustration of exponential decay
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
        Pads the list with -1 to make its length n
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

def get_causal_matrix(N):
    '''
        Return lower triangular matrix of size NxN
    '''
    causal_matrix = torch.ones(N, N)
    causal_matrix = torch.tril(causal_matrix)
    return causal_matrix
   
def get_qk_causal_matrix(N, num_deletes = 10):
    '''
        We return a lower triangular ones matrix of size (N ) x (N ) then
        we select num_deletes columns and rows randomly to delete.
    '''
    causal_matrix = torch.ones(N, N)
    causal_matrix = torch.tril(causal_matrix)
    # Select num_deletes number from range (0, N+num_deletes)
    indices_col = np.random.choice(range(1, N - 1), size=N - num_deletes - 2, replace=False)
    indices_row = np.random.choice(range(1, N -1), size=N - num_deletes - 2, replace=False)

    # sort in ascending order
    indices_col.sort()
    indices_row.sort()

    indices_col_final = torch.tensor([0] + list(indices_col) + [N  - 1])
    indices_row_final = torch.tensor([0] + list(indices_row) + [N- 1])
    
    # Delete the selected columns and rows from the causal_matrix
    causal_matrix = torch.index_select(causal_matrix, dim=0, index=torch.tensor(indices_row_final))
    causal_matrix = torch.index_select(causal_matrix, dim=1, index=torch.tensor(indices_col_final))
    
    return causal_matrix

def get_subtracted_list(N, n_min = 10, n_max = 20):
    '''
        In a loop we select a random range in the range (n_min, n_max) and subtract it from N.
        We add the subtracted value to a list and return it.
    '''
    subtracted_list = []
    while N > n_max:
        n = np.random.randint(n_min, n_max)
        subtracted_list.append(n)
        N -= n
    subtracted_list.append(N)
    return subtracted_list

def get_hash_causal_matrix(N, n_min = 10, n_max = 20):

    subtracted_list = get_subtracted_list(N, n_min, n_max)
    output_mat = torch.zeros(N, N)
    start_idx= 0
    for num in subtracted_list:
        output_mat[start_idx:start_idx+num, start_idx:start_idx+num] = 1
        start_idx += num
    output_mat = torch.tril(output_mat)
    return output_mat

def return_all_ones(inp_matrx):
    '''
        This function returns a one if all the elements in the input matrix are 1 
    '''
    return torch.all(inp_matrx == 1)

def get_num_ones_and_offset(matrix, BLKSZE_I, BLKSZE_J):

    num_ones = torch.zeros((matrix.shape[0]//BLKSZE_I,))
    offset = torch.zeros((matrix.shape[0]//BLKSZE_I,))

    for i in range(0, matrix.shape[0], BLKSZE_I):
        all_one_found = False
        for j in range(0, matrix.shape[1], BLKSZE_J):
            cur_out = return_all_ones(matrix[i:i+BLKSZE_I, j:j+BLKSZE_J])
            num_ones[i//BLKSZE_I] += int(cur_out)
            if not all_one_found and int(cur_out):
                offset[i//BLKSZE_I] = j//BLKSZE_J
                all_one_found = True
        print("Finished row", i//BLKSZE_I)

    return num_ones, offset

def get_int32_from_binaryMat(matrix):
    # Convert the matrix to a numpy array
    matrix = matrix.numpy()
    # Prepare a list to hold the resulting 32-bit integers
    int32_array = []

    # Iterate over each row and process it
    for row in range(matrix.shape[0]):
        # Read the row
        row_bits = matrix[row, :]
        
        # Initialize a list to hold 32-bit integers for this row
        row_int32_list = []
        
        # Process the column in chunks of 32 bits
        for i in range(0, len(row_bits), 32):
            # Get a chunk of 32 bits, if the chunk is less than 32 bits, pad with zeros
            chunk = row_bits[i:i+32]
            if len(chunk) < 32:
                chunk = np.pad(chunk, (0, 32-len(chunk)), 'constant')
            
            # Convert the chunk to a 32-bit integer
            int32_val = int(''.join(chunk.astype(str)), 2)
            
            # Append the integer to the column's list
            row_int32_list.append(int32_val)
        
        # Append the list of 32-bit integers for this column to the main list
        int32_array.append(row_int32_list)

    # Convert the list of lists into a numpy array
    # int32_array = np.array(int32_array, dtype=np.uint32)
    int32_array = torch.tensor(int32_array, dtype=torch.int32)
    return int32_array

def get_16bit_from_binaryMat(matrix):
    # Convert the matrix to a numpy array
    matrix = matrix.numpy()
    # Prepare a list to hold the resulting 32-bit integers
    bit16_arr = []

    # Iterate over each row and process it
    for row in range(matrix.shape[0]):
        # Read the row
        row_bits = matrix[row, :]
        
        # Initialize a list to hold 16-bit integers for this row
        row_16bit_list = []
        
        # Process the column in chunks of 16 bits
        for i in range(0, len(row_bits), 16):
            # Get a chunk of 32 bits, if the chunk is less than 16 bits, pad with zeros
            chunk = row_bits[i:i+16]
            if len(chunk) < 16:
                chunk = np.pad(chunk, (0, 16-len(chunk)), 'constant')
            
            # Convert the chunk to a 16-bit integer
            bit16_val = int(''.join(chunk.astype(str)), 2)
            
            # Append the integer to the column's list
            row_16bit_list.append(bit16_val)
        
        # Append the list of 32-bit integers for this column to the main list
        bit16_arr.append(row_16bit_list)

    # Convert the list of lists into a numpy array
    # int32_array = np.array(int32_array, dtype=np.uint32)
    bit16_arr = torch.tensor(bit16_arr, dtype=torch.int16)
    return bit16_arr