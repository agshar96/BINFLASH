import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
sys.path.append('/home/agniv/Documents/BINFLASH/')

import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from utils import create_symmetric_sparse_matrix, plot_binary_matrix
from triton_kernels.binBlkMask_kernels import return_binBlk_matrices
dtype = torch.float16
device = 'cuda'
for i in range(8,15):

    print(f'Starting with {2**i}...')
    N_CTX = 2**i
    sparse_mat = create_symmetric_sparse_matrix(N_CTX, base_prob=0.5, decay=0.3, pow = i)
    # maskMat_sparse = torch.tensor(sparse_mat, dtype=dtype, device=device)
    # binBlkmat_sparse = return_binBlk_matrices(maskMat_sparse, 128, 32)
    # sparse_mat = np.eye(N_CTX)
    # sparse_mat = np.tril(np.ones((N_CTX, N_CTX)))
    # sparse_mat = np.ones((N_CTX, N_CTX))
    print("Matrix created")
    plot_binary_matrix(sparse_mat)

    sparse_mat_csr = csr_matrix(sparse_mat)
    print("CSR complete")

    rcm_order = reverse_cuthill_mckee(sparse_mat_csr)
    print("RCM complete")

    # Reorder the grid according to RCM ordering
    reordered_grid_RCM = sparse_mat_csr[rcm_order][:, rcm_order].toarray()
    print("Reordering complete")
    plot_binary_matrix(reordered_grid_RCM)
    # maskMat_RCM = torch.tensor(reordered_grid_RCM, dtype=dtype, device=device)
    # binBlkMat_RCM = return_binBlk_matrices(maskMat_RCM, 128, 32)

    # print(binBlkmat_sparse.sum())
    # print(binBlkMat_RCM.sum())

    # np.savez(f'saved_mats/RCM/RCM_mats_{N_CTX}.npz', maskMat = sparse_mat, maskMat_RCM = reordered_grid_RCM)
    print(f'Finished with {2**i}...')