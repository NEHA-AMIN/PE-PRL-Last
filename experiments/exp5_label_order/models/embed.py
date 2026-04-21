import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        # === EXPERIMENT 5: LABEL + ORDER ===
        # Import both components
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        # LABEL: Legendre polynomials (from Exp3)
        from legendre_embedding import LegendrePositionEmbedding
        self.legendre_embedding = LegendrePositionEmbedding(
            d_model=d_model, 
            max_len=5000, 
            scaling=True
        )
        
        # ORDER: Signed displacements (from Exp4)
        from ordering_operator import OrderingOperator
        self.ordering_operator = OrderingOperator()
        # === END EXPERIMENT 5 ===

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # === EXPERIMENT 5: LABEL + ORDER (Equations 1 + 3) - CORRECTED ===
        # X'_i = X_i + T_i + P_i + O_i
        #
        # CRITICAL FIX: Order operator must work in POSITIONAL space (on Legendre),
        # not SEMANTIC space (on values), to create purely positional signal.
        # This ensures Label + Order are both positional components.
        #
        # NO distance decay (α = 1, uniform weighting)
        # NO feature-space weighting (w_ij removed)
        # YES temporal (for fair comparison)
        
        value_emb = self.value_embedding(x)  # [B, L, D] - Semantic content
        temporal_emb = self.temporal_embedding(x_mark)  # [B, L, D] - Time features
        
        # LABEL component (Equation 1): Orthogonal distinctiveness
        legendre_pos = self.legendre_embedding(x)  # [B, L, D] - Positional labels
        
        # ORDER component (Equation 3): Signed displacements in POSITIONAL space
        # CRITICAL: Applied to legendre_pos (positional), NOT value_emb (semantic)
        # This computes: O_i = (1/(L-1)) · Σ_{j≠i} (P_i - P_j)
        # where P_i is the Legendre embedding (positional structure)
        ordering_pos = self.ordering_operator(legendre_pos)  # [B, L, D] - Positional order
        
        # Optional: Add scaling for numerical stability (matching Exp 2 approach)
        ordering_pos = ordering_pos / math.sqrt(value_emb.size(-1))
        
        # Combine all components (Equation 9)
        x = value_emb + temporal_emb + legendre_pos + ordering_pos
        # === END EXPERIMENT 5 ===
        
        return self.dropout(x)
