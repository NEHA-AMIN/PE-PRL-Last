import torch
import torch.nn as nn
import numpy as np
from scipy.special import legendre
import math


class LegendrePositionEmbedding(nn.Module):
    """
    Legendre Polynomial Position Embeddings (Equation 1 from paper).
    
    Provides LABEL component: Orthogonal distinctiveness for each position.
    
    For position i: P_i = [L_0(x_i), L_1(x_i), ..., L_{d-1}(x_i)]
    where L_k are Legendre polynomials and x_i ∈ [-1, 1].
    
    Satisfies: ⟨P_n, P_m⟩ = δ_{nm} (Kronecker delta)
    
    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length (default: 5000)
        scaling (bool): If True, scale by 1/sqrt(d_model) for stability
    """
    
    def __init__(self, d_model, max_len=5000, scaling=True):
        super(LegendrePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scaling = scaling
        
        # Pre-compute Legendre embeddings
        legendre_emb = self._generate_legendre_embeddings(max_len, d_model)
        
        # Register as buffer (not trainable)
        self.register_buffer('legendre_emb', legendre_emb)
    
    def _generate_legendre_embeddings(self, seq_len, d_model):
        """
        Generate Legendre polynomial embeddings.
        
        Maps positions [0, 1, ..., seq_len-1] to [-1, 1] (standard Legendre domain)
        Evaluates L_0(x), L_1(x), ..., L_{d-1}(x) at each position.
        
        Returns:
            torch.Tensor: [seq_len, d_model] embeddings
        """
        # Map positions to [-1, 1]
        if seq_len == 1:
            positions = np.array([0.0])
        else:
            positions = 2.0 * np.arange(seq_len) / (seq_len - 1) - 1.0
        
        # Initialize embedding matrix
        P = np.zeros((seq_len, d_model))
        
        # Evaluate each Legendre polynomial L_k(x) at all positions
        for k in range(d_model):
            L_k = legendre(k)  # k-th Legendre polynomial
            P[:, k] = L_k(positions)
        
        # Convert to tensor
        P = torch.FloatTensor(P)
        
        # Scale by 1/sqrt(d_model) for numerical stability
        if self.scaling:
            P = P / math.sqrt(d_model)
        
        return P
    
    def forward(self, x):
        """
        Get Legendre position embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, ...]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, d_model] position embeddings
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Check if sequence is longer than pre-computed
        if seq_len > self.max_len:
            print(f"Warning: seq_len ({seq_len}) > max_len ({self.max_len}). Regenerating.")
            legendre_emb = self._generate_legendre_embeddings(seq_len, self.d_model)
            legendre_emb = legendre_emb.to(x.device)
        else:
            legendre_emb = self.legendre_emb[:seq_len, :]
        
        # Expand for batch: [seq_len, d_model] -> [batch_size, seq_len, d_model]
        return legendre_emb.unsqueeze(0).expand(batch_size, -1, -1)

# Made with Bob
