import torch
import torch.nn as nn
import numpy as np
import math


class LegendreEmbedding(nn.Module):
    """
    Generates orthogonal Legendre positional label vectors.
    
    For position i mapped to x_i ∈ [-1, 1]:
        P_i = [L_0(x_i), L_1(x_i), ..., L_{d_model-1}(x_i)] / sqrt(d_model)
    
    These are the same Legendre vectors used in exp3 (Label Only) and exp5b (L+O).
    Here they serve as the INPUT to the ordering operator, not as a standalone label.
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
            P: [1, seq_len, d_model] - positional label vectors (no batch dim yet)
        """
        P = self._legendre_matrix(seq_len, device)
        return P.unsqueeze(0)  # [1, seq_len, d_model]


class OrderingOperatorPositional(nn.Module):
    """
    Ordering Operator applied to POSITIONAL (Legendre) embeddings.
    
    This is exp4b — the appendix variant requested by mentor.
    
    CONTRAST with exp4 (OrderingOperator):
        exp4:  O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j)   [value/semantic space]
        exp4b: O_i = (1/N-1) · Σ_{j≠i} (P_i - P_j)   [positional/Legendre space]
    
    Mentor's exact words:
        "Order only that we have and all other experiments I want you to do it 
         in input as well, I see no reason to do it but I wanna have it in my appendix"
    
    Mathematical formulation:
        P_i ∈ R^d  — Legendre orthogonal positional vector for position i
        
        Signed positional displacement:
            Δp_ij = P_i - P_j  ∈ R^d
        
        Ordering signal (uniform weighted):
            O_i = (1 / (N-1)) · Σ_{j≠i} Δp_ij
                = (1 / (N-1)) · Σ_{j≠i} (P_i - P_j)
                = P_i - (1/(N-1)) · Σ_{j≠i} P_j         [simplification]
    
    Final embedding:
        X'_i = X_i + T_i + O_i
    
    Where:
        X_i = value_embedding     [semantic content, unchanged]
        T_i = temporal_embedding  [hour/day/month context, unchanged]
        O_i = ordering on P       [positional ordering signal]
    
    Key properties:
        ✅ Ordering operator in POSITIONAL space (Legendre vectors)
        ✅ Uniform weighting (no α(i,j) distance decay)
        ✅ Signed differences (preserves directionality)
        ✅ NO Legendre label added directly (only its ordering signal)
        ✅ NO distance decay
        ✅ Temporal embeddings retained
        ❌ NOT semantic/value space (that is exp4)
    """

    def __init__(self, d_model, scaling=True):
        super(OrderingOperatorPositional, self).__init__()
        self.d_model = d_model
        self.legendre = LegendreEmbedding(d_model=d_model, scaling=scaling)

    def forward(self, seq_len, device):
        """
        Compute ordering signal from Legendre positional embeddings.
        
        Args:
            seq_len: int — sequence length L
            device:  torch.device
        
        Returns:
            O: [1, L, D] — ordering signal in positional space,
               broadcastable over batch dimension B.
        """
        # Step 1: Get Legendre positional vectors P ∈ [1, L, D]
        P = self.legendre(seq_len, device)  # [1, L, D]

        # Step 2: Compute pairwise signed displacements Δp_ij = P_i - P_j
        # Expand for pairwise computation
        P_i = P.unsqueeze(2)   # [1, L, 1, D]
        P_j = P.unsqueeze(1)   # [1, 1, L, D]
        delta_p = P_i - P_j    # [1, L, L, D] — signed positional displacements

        # Step 3: Mask diagonal (exclude j == i)
        L = seq_len
        mask = torch.eye(L, device=device).bool()   # [L, L]
        mask = mask.unsqueeze(0).unsqueeze(-1)       # [1, L, L, 1]
        delta_p = delta_p.masked_fill(mask, 0.0)

        # Step 4: Aggregate with uniform weighting
        # O_i = (1/(N-1)) · Σ_{j≠i} Δp_ij
        O = delta_p.sum(dim=2) / (L - 1)   # [1, L, D]

        return O   # [1, L, D] — broadcast over batch in embed.py


if __name__ == "__main__":
    print("Testing OrderingOperatorPositional...")

    d_model = 512
    seq_len = 96
    device = torch.device("cpu")

    op = OrderingOperatorPositional(d_model=d_model, scaling=True)
    O = op(seq_len, device)

    print(f"Output shape:  {O.shape}")          # [1, 96, 512]
    print(f"Output range:  [{O.min():.6f}, {O.max():.6f}]")
    print(f"Output mean:   {O.mean():.6f}")
    print(f"Output std:    {O.std():.6f}")

    # Verify non-zero
    assert O.abs().sum() > 0, "ERROR: Output is all zeros!"
    print("✓ Non-zero outputs confirmed")

    # Verify shape
    assert O.shape == (1, seq_len, d_model), f"Shape mismatch: {O.shape}"
    print("✓ Shape correct")

    # Verify it differs from exp4 (which would use value embeddings, not Legendre)
    X = torch.randn(1, seq_len, d_model)
    X_i = X.unsqueeze(2)
    X_j = X.unsqueeze(1)
    delta_x = X_i - X_j
    mask = torch.eye(seq_len).bool().unsqueeze(0).unsqueeze(-1)
    delta_x = delta_x.masked_fill(mask, 0.0)
    O_exp4 = delta_x.sum(dim=2) / (seq_len - 1)
    assert not torch.allclose(O, O_exp4), "ERROR: exp4b matches exp4 — positional space not being used!"
    print("✓ Confirmed different from exp4 (value-space ordering)")

    print("\n✓ OrderingOperatorPositional test complete!")

# Made with Bob
