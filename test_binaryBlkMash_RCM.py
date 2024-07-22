from utils import create_blk_one_matrix, plot_binary_matrix, create_symmetric_sparse_matrix
import torch
from blockmask_attention import attention_binaryBlkMat
# from customMask_attention import attention_bcwrd_masked
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


torch.manual_seed(20)
Z = 1
H = 1
N_CTX = 1024
HEAD_DIM = 64
dtype=torch.float16
causal = False
torch.manual_seed(20)
q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

#### Original maskMat
maskMat = torch.from_numpy(create_symmetric_sparse_matrix(N_CTX, base_prob=0.5, decay=0.1))

#### OPTIMIZING USING RCM
sparse_mat_csr = csr_matrix(maskMat)

rcm_order = reverse_cuthill_mckee(sparse_mat_csr)
print("RCM complete")

# Reorder the grid according to RCM ordering
reordered_grid_RCM = sparse_mat_csr[rcm_order][:, rcm_order].toarray()
RCM_mat_blk = create_blk_one_matrix(reordered_grid_RCM, 128, 32)

# plot_binary_matrix(reordered_grid_RCM)
# plot_binary_matrix(RCM_mat_blk)
reordered_grid_RCM = torch.from_numpy(reordered_grid_RCM)
reordered_grid_RCM = reordered_grid_RCM.to(dtype=dtype).to("cuda")
RCM_mat_blk = RCM_mat_blk.to(dtype=dtype).to("cuda")
###### OPTMIZATION DONE

sm_scale = 0.5
dout = torch.randn_like(q)

# reference implementation
p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
if True:
    p[:, :, reordered_grid_RCM == 0] = float("-inf")
p = torch.softmax(p.float(), dim=-1).half()
# p = torch.exp(p)
ref_out = torch.matmul(p, v)
ref_out.backward(dout)
ref_dv, v.grad = v.grad.clone(), None
ref_dk, k.grad = k.grad.clone(), None
ref_dq, q.grad = q.grad.clone(), None

tri_out = attention_binaryBlkMat(q, k, v, causal, sm_scale, reordered_grid_RCM, RCM_mat_blk, RCM_mat_blk).half()
print(torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0))

tri_out.backward(dout)
tri_dv, v.grad = v.grad.clone(), None
tri_dk, k.grad = k.grad.clone(), None
tri_dq, q.grad = q.grad.clone(), None

print(torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0))
print(torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0))
print(torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0))
