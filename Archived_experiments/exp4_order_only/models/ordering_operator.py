import torch
import torch.nn as nn


class OrderingOperator(nn.Module):
    """
    Pure Ordering Operator - Tests directional signal only (Equation 3).
    
    Computes: O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j)
    
    This captures relative positioning through signed feature-space 
    displacements WITHOUT distance-based weighting.
    
    - Uses signed differences (preserves directionality)
    - NO index-based decay α(i,j)
    - NO feature-space weighting w_ij
    - NO Legendre labels
    
    Pure test of whether ordering information alone helps.
    """
    
    def __init__(self):
        super(OrderingOperator, self).__init__()
    
    def forward(self, X):
        """
        Compute pure ordering signal.
        
        Args:
            X: [B, L, D] - input embeddings (token embeddings)
        
        Returns:
            O: [B, L, D] - ordering-based representations
        """
        B, L, D = X.shape
        device = X.device
        
        # 1. Compute SIGNED feature-space displacements: Δx_ij = X_i - X_j
        X_i = X.unsqueeze(2)  # [B, L, 1, D]
        X_j = X.unsqueeze(1)  # [B, 1, L, D]
        delta_x = X_i - X_j   # [B, L, L, D] - signed differences
        
        # 2. Mask diagonal (exclude self: j ≠ i)
        mask = torch.eye(L, device=device).bool()  # [L, L]
        mask = mask.unsqueeze(0).unsqueeze(-1)     # [1, L, L, 1]
        delta_x = delta_x.masked_fill(mask, 0.0)
        
        # 3. Aggregate with UNIFORM weighting (no distance bias)
        # O_i = (1/N-1) · Σ_{j≠i} Δx_ij
        O = delta_x.sum(dim=2) / (L - 1)  # [B, L, D]
        
        return O


if __name__ == "__main__":
    # Test ordering operator
    print("Testing Ordering Operator...")
    
    B, L, D = 32, 96, 512
    ordering_op = OrderingOperator()
    
    # Create dummy input
    X = torch.randn(B, L, D)
    
    # Get ordering signal
    O = ordering_op(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {O.shape}")
    print(f"Output range: [{O.min():.4f}, {O.max():.4f}]")
    print(f"Output mean: {O.mean():.4f}, std: {O.std():.4f}")
    
    # Verify output is not all zeros
    if O.abs().sum() > 0:
        print("✓ Ordering operator producing non-zero outputs")
    else:
        print("✗ Warning: Outputs are all zeros!")
    
    print("\n✓ Ordering operator test complete!")
