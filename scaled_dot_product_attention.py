import torch
import torch.nn as nn
from Transformer.utils.mask import get_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, masked=False, device='cuda'):
        super().__init__()
        self.scale = 1 / (d_model ** 0.5)
        self.masked = masked
        self.device = device

    def forward(self, Q, K, V):
        Q_matmal_K = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if self.masked:
            mask = get_mask(Q.shape[-2]).to(self.device)
            Q_matmal_K = Q_matmal_K + mask
        score = torch.softmax(Q_matmal_K, dim=-1)
        attention = torch.matmul(score, V)
        return attention
