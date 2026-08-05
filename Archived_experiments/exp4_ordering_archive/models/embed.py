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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
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
    """Zhou's original DataEmbedding — unchanged."""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                                   if embed_type != 'timeF'
                                   else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_ordering_sem(nn.Module):
    """
    Experiment: ordering_new_sem_space
    
    Formula:
        X'_i = X_i + T_i + O_i^sem

    where:
        delta_i^val  = 0                if i = 0
                       X_i - X_{i-1}   if i >= 1
        x_bar^val    = (1/N) * sum_i ||X_i||_2   (scalar per batch element)
        O_i^sem      = delta_i^val / (x_bar^val + 1e-8)

    Components included:
        value embedding    YES  (X_i)
        temporal embedding YES  (T_i)
        ordering signal    YES  (O_i^sem, built in semantic space)
    Components excluded:
        sinusoidal PositionalEmbedding  NO
        Legendre embedding              NO

    Properties:
        - Scale-invariant: multiplying all X_i by alpha leaves O_i unchanged
        - Translation-invariant: adding constant c to all X_i leaves delta unchanged
        - Locally order-sensitive: permuting input changes O_i
        - delta[:, 0, :] == 0  (zero-pad boundary condition)
        - No learnable parameters beyond TokenEmbedding and temporal embedding
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_ordering_sem, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = (TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                                   if embed_type != 'timeF'
                                   else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq))
        self.dropout = nn.Dropout(p=dropout)
        self._diag_step = 0   # diagnostic counter — prints first batch of each epoch only

    def forward(self, x, x_mark):
        # ── standard components ───────────────────────────────────────────────
        val  = self.value_embedding(x)          # [B, L, D]   X_i
        temp = self.temporal_embedding(x_mark)  # [B, L, D]   T_i

        # ── consecutive delta in semantic space ───────────────────────────────
        delta = torch.zeros_like(val)                        # delta_0 = 0  [B, L, D]
        delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]   # delta_i = X_i - X_{i-1}

        # ── scalar normalization: mean of L2 norms over sequence ──────────────
        # val.norm(dim=-1): [B, L]
        # .mean(dim=1, keepdim=True): [B, 1]
        # .unsqueeze(-1): [B, 1, 1]
        x_bar = val.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]

        ordering = delta / (x_bar + 1e-8)       # [B, L, D]   O_i^sem

        # ── diagnostic: signal magnitude — first batch of each epoch only ────
        if self.training and self._diag_step % 100 == 0:
            with torch.no_grad():
                delta_norm = delta.norm(dim=-1).mean().item()
                xbar_val   = x_bar.mean().item()
                order_norm = ordering.norm(dim=-1).mean().item()
                val_norm   = val.norm(dim=-1).mean().item()
                print(f"[SEM step={self._diag_step}] "
                      f"delta_norm={delta_norm:.4f} | x_bar={xbar_val:.4f} | "
                      f"order_norm={order_norm:.4f} | val_norm={val_norm:.4f} | "
                      f"ratio={order_norm/val_norm:.4f}")
        if self.training:
            self._diag_step += 1

        return self.dropout(val + temp + ordering)           # [B, L, D]
