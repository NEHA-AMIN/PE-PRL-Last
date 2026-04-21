import torch
import torch.nn as nn
import math


class LegendreEmbedding(nn.Module):
    """
    Generates orthogonal Legendre positional label vectors.
    
    For position i mapped to x_i ∈ [-1, 1]:
        P_i = [L_0(x_i), L_1(x_i), ..., L_{d_model-1}(x_i)] / sqrt(d_model)
    
    These vectors are used in exp4b to compute consecutive deltas Δp_i = P_i - P_{i-1}
    in the value matrix of the attention mechanism.
    """

    def __init__(self, d_model, max_len=5000, scaling=True):
        super(LegendreEmbedding, self).__init__()
        self.d_model = d_model
        self.scaling = scaling
        self.max_len = max_len

    def _legendre_matrix(self, seq_len, device):
        """
        Compute Legendre polynomial matrix of shape [seq_len, d_model].
        
        Positions mapped: i -> x_i = 2i/(seq_len-1) - 1, x_i in [-1, 1]
        """
        # Map position indices to [-1, 1]
        if seq_len == 1:
            positions = torch.zeros(1, device=device)
        else:
            positions = 2.0 * torch.arange(seq_len, dtype=torch.float32, device=device) / (seq_len - 1) - 1.0

        # Build Legendre matrix using recurrence:
        # L_0(x) = 1
        # L_1(x) = x
        # L_n(x) = ((2n-1)*x*L_{n-1}(x) - (n-1)*L_{n-2}(x)) / n
        P = torch.zeros(seq_len, self.d_model, device=device)
        if self.d_model >= 1:
            P[:, 0] = 1.0
        if self.d_model >= 2:
            P[:, 1] = positions
        for n in range(2, self.d_model):
            P[:, n] = ((2 * n - 1) * positions * P[:, n - 1] - (n - 1) * P[:, n - 2]) / n

        if self.scaling:
            P = P / math.sqrt(self.d_model)

        return P  # [seq_len, d_model]

    def forward(self, seq_len, device):
        """
        Returns Legendre positional vectors for a sequence of given length.
        
        Returns:
            P: [1, seq_len, d_model] - positional label vectors (batch dim = 1)
        """
        P = self._legendre_matrix(seq_len, device)
        return P.unsqueeze(0)  # [1, seq_len, d_model]


if __name__ == "__main__":
    print("Testing LegendreEmbedding...")

    d_model = 512
    seq_len = 96
    device = torch.device("cpu")

    legendre = LegendreEmbedding(d_model=d_model, scaling=True)
    P = legendre(seq_len, device)

    print(f"Output shape:  {P.shape}")          # [1, 96, 512]
    print(f"Output range:  [{P.min():.6f}, {P.max():.6f}]")
    print(f"Output mean:   {P.mean():.6f}")
    print(f"Output std:    {P.std():.6f}")

    # Verify non-zero
    assert P.abs().sum() > 0, "ERROR: Output is all zeros!"
    print("✓ Non-zero outputs confirmed")

    # Verify shape
    assert P.shape == (1, seq_len, d_model), f"Shape mismatch: {P.shape}"
    print("✓ Shape correct")

    # Test consecutive deltas (what will be used in attention)
    delta_p = torch.zeros_like(P)
    delta_p[:, 1:, :] = P[:, 1:, :] - P[:, :-1, :]
    print(f"\nDelta P shape: {delta_p.shape}")
    print(f"Delta P[0] (should be zero): {delta_p[:, 0, :].abs().sum():.6f}")
    print(f"Delta P[1:] non-zero: {delta_p[:, 1:, :].abs().sum() > 0}")

    print("\n✓ LegendreEmbedding test complete!")

# Made with Bob
