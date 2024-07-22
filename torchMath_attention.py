import torch

class _torchMath_attention(torch.nn.Module):

    def forward(self, q, k, v, sm_scale, maskMat = None):
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if maskMat is not None:
            p[:, :, maskMat == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        return torch.matmul(p, v)
    
torchMath_attention = _torchMath_attention()
    
