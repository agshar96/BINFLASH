from utils import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, depth_first_order
import torch
import numpy as np

for i in range(8,15):

    print(f'Starting with {2**i}...')
    N_CTX = 2**i
    # sparse_mat = create_symmetric_sparse_matrix(N_CTX, base_prob=0.5, decay=0.3, pow = i)
    # sparse_mat = get_hash_causal_matrix(N_CTX, 32, 128)
    sparse_mat = get_causal_matrix(N_CTX)
    # sparse_mat = np.eye(N_CTX)
    # sparse_mat = np.tril(np.ones((N_CTX, N_CTX)))
    # sparse_mat = np.ones((N_CTX, N_CTX))
    print("Matrix created")
    # plot_binary_matrix(sparse_mat)

    ## These are required for all BinBlockMasks and derived kernels
    sparse_mat_blk = create_blk_one_matrix(sparse_mat, 128,32)
    sparse_mat_blk_T = create_blk_one_matrix(sparse_mat.T, 128,32)

    ## These are changes for offset binBlkMask
    num_ones, offset = get_num_ones_and_offset(sparse_mat, 128, 32)
    num_ones_T, offset_T = get_num_ones_and_offset(sparse_mat.T, 128, 32)


    ## These are changes for int32 binBlkMask
    sparse_mat_int16 = get_16bit_from_binaryMat(sparse_mat_blk.to(dtype=torch.uint8))
    sparse_mat_int16_T = get_16bit_from_binaryMat(sparse_mat_blk_T.to(dtype=torch.uint8))

    torch.save({'maskMat': sparse_mat, 'maskMat_blk': sparse_mat_blk, 'maskMat_blk_T': sparse_mat_blk_T,
                'num_ones': num_ones, 'offset': offset, 'num_ones_T': num_ones_T, 'offset_T': offset_T,
                'maskMat_int16': sparse_mat_int16, 'maskMat_int16_T': sparse_mat_int16_T}, 
                f'saved_mats/causal/mat_n_blks_{N_CTX}.pt')
    print(f'Finished with {2**i}...')
    # sparse_mat_csr = csr_matrix(sparse_mat)
    # print("CSR complete")

    # rcm_order = reverse_cuthill_mckee(sparse_mat_csr)
    # print("RCM complete")

    # # Reorder the grid according to RCM ordering
    # reordered_grid_RCM = sparse_mat_csr[rcm_order][:, rcm_order].toarray()
    # RCM_mat_blk = create_blk_one_matrix(reordered_grid_RCM, 128, 32)
    # RCM_mat_blk_T = create_blk_one_matrix(reordered_grid_RCM.T, 128, 32)

    
    # plot_binary_matrix(reordered_grid_RCM)
    # plot_binary_matrix(sparse_mat_blk)
    # plot_binary_matrix(RCM_mat_blk)
    # print(sparse_mat_blk.sum())
    # print(RCM_mat_blk.sum())

    # np.savez(f'saved_mats/ones/mat_n_blks_{N_CTX}.npz', maskMat = sparse_mat, maskMat_RCM = reordered_grid_RCM, maskMat_blk = sparse_mat_blk.numpy(), maskMat_RCM_blk = RCM_mat_blk.numpy(),
    #          maskMat_blk_T = sparse_mat_blk_T.numpy(), maskMat_RCM_blk_T = RCM_mat_blk_T.numpy())
    # print(f'Finished with {2**i}...')


