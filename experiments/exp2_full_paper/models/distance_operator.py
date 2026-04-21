import torch
import torch.nn as nn
import torch.nn.functional as F


class DistancePositionOperator(nn.Module):
    """
    Implements distance-based position encoding from paper.
    
    Full formulation:
        O_i = Σ_{j∈[N]\{i}} α(i,j) · (w_ij ⊙ Δx_ij)
    
    where:
        - Δx_ij = X_i - X_j  (signed feature-space displacement)
        - α(i,j) = 1 / (1 + |i-j|^a)  (index-based distance decay)
        - w_ij = 1 / (1 + d_ij)  (feature-space weighting)
        - d_ij = L1 or L2 distance between X_i and X_j
        - ⊙ denotes element-wise multiplication
    
    Args:
        decay_a (float): Decay parameter for index-based distance (default: 1.0)
        distance_type (str): Type of feature-space distance ('l1' or 'l2')
    """
    
    def __init__(self, decay_a=1.0, distance_type='l1'):
        super(DistancePositionOperator, self).__init__()
        self.decay_a = decay_a
        self.distance_type = distance_type
    
    def forward(self, X):
        """
        Compute distance-based position representations.
        
        Args:
            X: [B, L, D] - input embeddings (token embeddings)
        
        Returns:
            O: [B, L, D] - distance-based position representations
        """
        B, L, D = X.shape
        device = X.device
        
        # 1. Compute SIGNED feature-space displacements: Δx_ij = X_i - X_j
        # Expand dimensions for pairwise computation
        X_i = X.unsqueeze(2)  # [B, L, 1, D]
        X_j = X.unsqueeze(1)  # [B, 1, L, D]
        delta_x = X_i - X_j   # [B, L, L, D] - SIGNED differences (preserves direction)
        
        # 2. Compute index-based distance decay: α(i,j) = 1 / (1 + |i-j|^a)
        i_idx = torch.arange(L, device=device).unsqueeze(1)  # [L, 1]
        j_idx = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        index_dist = torch.abs(i_idx - j_idx).float()  # [L, L]
        alpha = 1.0 / (1.0 + index_dist ** self.decay_a)  # [L, L]
        
        # 3. Compute feature-space distances d_ij
        if self.distance_type == 'l1':
            # L1 distance: Σ_k |x_i,k - x_j,k|
            d_ij = torch.abs(delta_x).sum(dim=-1)  # [B, L, L]
        elif self.distance_type == 'l2':
            # L2 distance: sqrt(Σ_k (x_i,k - x_j,k)²)
            d_ij = torch.sqrt((delta_x ** 2).sum(dim=-1) + 1e-8)  # [B, L, L]
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}")
        
        # 4. Compute feature-space weighting: w_ij = 1 / (1 + d_ij)
        w_ij = 1.0 / (1.0 + d_ij)  # [B, L, L]
        
        # 5. Combine weights: α(i,j) · w_ij
        # alpha: [L, L] -> broadcast to [B, L, L]
        combined_weight = alpha.unsqueeze(0) * w_ij  # [B, L, L]
        
        # 6. Element-wise multiplication: (α(i,j) · w_ij) ⊙ Δx_ij
        # combined_weight: [B, L, L] -> [B, L, L, 1] for broadcasting
        # delta_x: [B, L, L, D]
        weighted_delta = combined_weight.unsqueeze(-1) * delta_x  # [B, L, L, D]
        
        # 7. Aggregate: O_i = Σ_j (α(i,j) · w_ij ⊙ Δx_ij)
        # Exclude self (j ≠ i) by masking diagonal
        mask = torch.eye(L, device=device).bool()  # [L, L]
        mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, L, L, 1]
        weighted_delta = weighted_delta.masked_fill(mask, 0.0)
        
        # Sum over j dimension
        O = weighted_delta.sum(dim=2)  # [B, L, D]
        
        return O
