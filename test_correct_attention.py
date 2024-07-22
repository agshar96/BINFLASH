import torch
from correct_fused_attention import attention_correct
from torchMath_attention import torchMath_attention

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
maskMat = torch.ones((N_CTX, N_CTX), dtype=dtype, device="cuda")
maskMat = maskMat.to(dtype=dtype).to("cuda")

sm_scale = 0.5
dout = torch.randn_like(q)

# reference implementation
ref_out = torchMath_attention.forward(q, k, v, sm_scale, maskMat)
ref_out.backward(dout)
ref_dv, v.grad = v.grad.clone(), None
ref_dk, k.grad = k.grad.clone(), None
ref_dq, q.grad = q.grad.clone(), None

# TRITON IMPLEMENTATION
tri_out = attention_correct(q, k, v, causal, sm_scale).half()
tri_out.backward(dout)
tri_dv, v.grad = v.grad.clone(), None
tri_dk, k.grad = k.grad.clone(), None
tri_dq, q.grad = q.grad.clone(), None

## Final testing
print(torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0))
print(torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0))
print(torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0))
print(torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0))