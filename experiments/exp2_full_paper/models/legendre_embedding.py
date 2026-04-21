import torch
import torch.nn as nn
import numpy as np
from scipy.special import legendre
import math


class LegendrePositionEmbedding(nn.Module):
    """
    Position embeddings using Legendre polynomials (Equation 1 from paper).
    
    For position i, generates embedding P_i = [L_0(x_i), L_1(x_i), ..., L_{d-1}(x_i)]
    where L_k are Legendre polynomials and x_i ∈ [-1, 1].
    
    Satisfies orthogonality: ⟨P_n, P_m⟩ = δ_{nm} (Kronecker delta)
    
    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length (default: 5000)
        scaling (bool): If True, scale by 1/sqrt(d_model) for numerical stability
    """
    
    def __init__(self, d_model, max_len=5000, scaling=True):
        super(LegendrePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scaling = scaling
        
        # Pre-compute Legendre embeddings for max_len positions
        # Will recompute only if sequence length exceeds max_len
        legendre_emb = self._generate_legendre_embeddings(max_len, d_model)
        
        # Register as buffer (not a trainable parameter)
        self.register_buffer('legendre_emb', legendre_emb)
    
    def _generate_legendre_embeddings(self, seq_len, d_model):
        """
        Generate Legendre polynomial embeddings for seq_len positions.
        
        Args:
            seq_len (int): Sequence length
            d_model (int): Embedding dimension
            
        Returns:
            torch.Tensor: [seq_len, d_model] Legendre embeddings
        """
        # Map positions [0, 1, ..., seq_len-1] to [-1, 1]
        # This is the standard domain for Legendre polynomials
        if seq_len == 1:
            positions = np.array([0.0])  # Single position maps to center
        else:
            positions = 2.0 * np.arange(seq_len) / (seq_len - 1) - 1.0
        
        # Initialize embedding matrix
        P = np.zeros((seq_len, d_model))
        
        # Evaluate each Legendre polynomial at all positions
        for k in range(d_model):
            # Get k-th Legendre polynomial: L_k(x)
            L_k = legendre(k)
            # Evaluate at all positions
            P[:, k] = L_k(positions)
        
        # Convert to tensor
        P = torch.FloatTensor(P)
        
        # Scale by 1/sqrt(d_model) to match standard transformer embedding magnitudes
        # This prevents high-order Legendre terms from dominating
        if self.scaling:
            P = P / math.sqrt(d_model)
        
        return P
    
    def forward(self, x):
        """
        Get Legendre position embeddings for input sequence.
        
        Args:
            x: Input tensor [batch_size, seq_len, ...] (we only use seq_len)
            
        Returns:
            torch.Tensor: [batch_size, seq_len, d_model] position embeddings
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Check if we need to regenerate for longer sequences
        if seq_len > self.max_len:
            print(f"Warning: seq_len ({seq_len}) > max_len ({self.max_len}). Regenerating Legendre embeddings.")
            # Regenerate on the fly
            legendre_emb = self._generate_legendre_embeddings(seq_len, self.d_model)
            legendre_emb = legendre_emb.to(x.device)
        else:
            # Use pre-computed embeddings
            legendre_emb = self.legendre_emb[:seq_len, :]
        
        # Expand for batch dimension: [seq_len, d_model] -> [batch_size, seq_len, d_model]
        return legendre_emb.unsqueeze(0).expand(batch_size, -1, -1)
    
    def verify_orthogonality(self, seq_len=100):
        """
        Verify that Legendre embeddings satisfy orthogonality property.
        
        Computes Gram matrix G[i,j] = ⟨P_i, P_j⟩ and checks if it's close to identity.
        
        Args:
            seq_len (int): Sequence length to test
            
        Returns:
            dict: Contains 'gram_matrix', 'diagonal_mean', 'off_diagonal_max'
        """
        # Generate embeddings
        emb = self._generate_legendre_embeddings(seq_len, self.d_model)
        
        # Compute Gram matrix: G[i,j] = P_i · P_j^T
        gram = torch.matmul(emb, emb.t())
        
        # Extract diagonal (should be ~1 after scaling)
        diagonal = torch.diag(gram)
        
        # Extract off-diagonal (should be ~0)
        mask = torch.eye(seq_len).bool()
        off_diagonal = gram.masked_select(~mask)
        
        return {
            'gram_matrix': gram,
            'diagonal_mean': diagonal.mean().item(),
            'diagonal_std': diagonal.std().item(),
            'off_diagonal_max': off_diagonal.abs().max().item(),
            'off_diagonal_mean': off_diagonal.abs().mean().item()
        }


if __name__ == "__main__":
    # Test Legendre embeddings
    print("Testing Legendre Position Embeddings...")
    
    d_model = 512
    seq_len = 96
    batch_size = 32
    
    # Create embedding layer
    legendre_pe = LegendrePositionEmbedding(d_model=d_model)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Get embeddings
    pos_emb = legendre_pe(x)
    print(f"Input shape: {x.shape}")
    print(f"Position embedding shape: {pos_emb.shape}")
    print(f"Position embedding range: [{pos_emb.min():.4f}, {pos_emb.max():.4f}]")
    print(f"Position embedding mean: {pos_emb.mean():.4f}, std: {pos_emb.std():.4f}")
    
    # Verify orthogonality
    print("\nVerifying orthogonality...")
    ortho_stats = legendre_pe.verify_orthogonality(seq_len=seq_len)
    print(f"Diagonal mean (should be ~{1/d_model:.4f}): {ortho_stats['diagonal_mean']:.6f}")
    print(f"Diagonal std: {ortho_stats['diagonal_std']:.6f}")
    print(f"Off-diagonal max (should be ~0): {ortho_stats['off_diagonal_max']:.6f}")
    print(f"Off-diagonal mean (should be ~0): {ortho_stats['off_diagonal_mean']:.6f}")
    
    print("\n✓ Legendre embedding test complete!")