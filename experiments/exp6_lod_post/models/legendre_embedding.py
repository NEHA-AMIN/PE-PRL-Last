import torch
import torch.nn as nn

class LegendrePositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, scaling=True):
        super(LegendrePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.scaling = scaling

    def forward(self, x):
        B, L, _ = x.shape
        device = x.device

        # Map positions to [-1, 1]
        if L == 1:
            positions = torch.zeros(1, device=device)
        else:
            positions = 2.0 * torch.arange(L, dtype=torch.float32, device=device) / (L - 1) - 1.0

        # Recurrence relation
        P = torch.zeros(L, self.d_model, device=device)
        if self.d_model >= 1:
            P[:, 0] = 1.0
        if self.d_model >= 2:
            P[:, 1] = positions
        for n in range(2, self.d_model):
            P[:, n] = ((2*n - 1) * positions * P[:, n-1] - (n-1) * P[:, n-2]) / n

        if self.scaling:
            P = P / (self.d_model ** 0.5)

        return P.unsqueeze(0).expand(B, -1, -1)  # [B, L, d_model]

# Made with Bob
