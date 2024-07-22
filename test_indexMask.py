import torch
from indexMask_attention import indexMask_attention

from utils import create_symmetric_sparse_matrix, plot_binary_matrix, create_stride_and_idx_mat

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
maskMat = torch.from_numpy(create_symmetric_sparse_matrix(N_CTX, base_prob=0.5, decay=0.1))
strideMat, idxMat = create_stride_and_idx_mat(maskMat, 128, 32)
# maskMat = torch.ones_like(maskMat)
maskMat = maskMat.to(dtype=dtype).to("cuda")

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

# size_bytes = 512
# triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)
tri_out = indexMask_attention(q, k, v, causal, sm_scale, maskMat, strideMat, idxMat).half()
# tri_out = attention(q, k, v, causal, sm_scale).half()
# tri_out.backward(dout)
# tri_dv, v.grad = v.grad.clone(), None
# tri_dk, k.grad = k.grad.clone(), None
# tri_dq, q.grad = q.grad.clone(), None
# # tri_out = attention(q, k, v, causal, sm_scale).half()

print(torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0))

print(torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0))
print(torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0))
print(torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0))