from utils import *
import torch
# from blockmask_attention import attention_binaryBlkMat
from customMask_attention import attention_masked
from binBlkMask_codes.bitShift_binBlkMask import attention_bitShift_binBlkMask
from binBlkMask_codes.combined_binBlkMask import attention_combined_binBlkMask
import numpy as np

torch.manual_seed(24)
Z = 1
H = 1
N_CTX = 1024
HEAD_DIM = 64
dtype=torch.float16
causal = True
torch.manual_seed(20)
q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

#### Original maskMat and binary_blk_mask
# maskMat = torch.from_numpy(create_symmetric_sparse_matrix(N_CTX, base_prob=0.5, decay=0.1))
# maskMat = torch.ones(1024,1024)
maskMat = get_causal_matrix(N_CTX)
# plot_binary_matrix(maskMat)
# num_ones, offset = get_num_ones_and_offset(maskMat, 128, 32)
# num_ones_T, offset_T = get_num_ones_and_offset(maskMat.T, 128, 32)
# # maskMat = torch.randint(0, 2, (1024, 1024))
# # maskMat = torch.randint(0, 2, (1024, 1024))
# binary_blk_mask = create_blk_one_matrix(maskMat, 128, 32)
# maskMat = torch.from_numpy(np.tril(np.ones((N_CTX, N_CTX))))
binary_blk_mask = torch.tensor(create_blk_one_matrix(maskMat, 128, 32), dtype=torch.uint8)
binary_blk_mask_T = torch.tensor(create_blk_one_matrix(maskMat.T, 128, 32), dtype=torch.uint8)
binary_blk_int32 = get_16bit_from_binaryMat(binary_blk_mask)
binary_blk_int32_T = get_16bit_from_binaryMat(binary_blk_mask_T)
binary_blk_int32 = binary_blk_int32.to(dtype=torch.int16).to("cuda")
binary_blk_int32_T = binary_blk_int32_T.to(dtype=torch.int16).to("cuda")
# binary_blk_mask = binary_blk_mask.to(dtype=dtype).to("cuda")
maskMat = maskMat.to(dtype=dtype).to("cuda")
# binary_blk_mask_T = binary_blk_mask_T.to(dtype=dtype).to("cuda")
# num_ones = num_ones.to(dtype=dtype).to("cuda")
# offset = offset.to(dtype=dtype).to("cuda")
# num_ones_T = num_ones_T.to(dtype=dtype).to("cuda")
# offset_T = offset_T.to(dtype=dtype).to("cuda")
# filename = "saved_mats_gen/mat_n_blks_" + str(N_CTX) + ".npz"
# data = np.load(filename)
# maskMat = torch.tensor(data['maskMat'], dtype=dtype, device="cuda")
# binary_blk_mask = torch.tensor(data['maskMat_blk'], dtype=dtype, device="cuda")

sm_scale = 0.5
dout = torch.randn_like(q)

# reference implementation
p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
if True:
    p[:, :, maskMat == 0] = float("-inf")
p = torch.softmax(p.float(), dim=-1).half()
# p = torch.exp(p)
ref_out = torch.matmul(p, v)
ref_out.backward(dout)
ref_dv, v.grad = v.grad.clone(), None
ref_dk, k.grad = k.grad.clone(), None
ref_dq, q.grad = q.grad.clone(), None

tri_out = attention_bitShift_binBlkMask(q, k, v, causal, sm_scale, maskMat, binary_blk_int32, binary_blk_int32_T).half()
tri_out.backward(dout)
tri_dv, v.grad = v.grad.clone(), None
tri_dk, k.grad = k.grad.clone(), None
tri_dq, q.grad = q.grad.clone(), None

print(torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0))

print(torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0))
print(torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0))
print(torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0))
