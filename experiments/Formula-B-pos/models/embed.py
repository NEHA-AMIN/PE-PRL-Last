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
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   padding_mode='circular')
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

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
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


# ─────────────────────────────────────────────────────────────────────────────
# Formula B — Global Mean Deviation Ordering in Positional Space
# ─────────────────────────────────────────────────────────────────────────────

class DataEmbedding_formula_b_pos(nn.Module):
    """Formula B: Global Mean Deviation Ordering in the Positional Pathway.

    Ordering formula
    ----------------
        P_i    = LegendrePositionEmbedding(x)     [B, L, D]   positional vectors

        mu_p   = (1/L) * sum_{k=1}^{L} P_k        [B, 1, D]   global mean of P
        delta_p = mu_p - P_i                       [B, L, D]   deviation from global mean

        p_bar  = (1/L) * sum_i ||P_i||_2           [B, 1, 1]   scalar normaliser
        ordering = delta_p / (p_bar + 1e-8)        [B, L, D]   normalised signal

    Output formula
    --------------
        X'_i = V_i + T_i + P_i + Ordering_i

    where:
        V_i = TokenEmbedding(x)                    semantic content
        T_i = TemporalEmbedding(x_mark)            calendar context
        P_i = LegendrePositionEmbedding(x)         positional label (added directly)
        Ordering_i = delta_p / (p_bar + 1e-8)      positional ordering signal

    Key distinction from Formula-A-pos
    ------------------------------------
        Formula A (positional): delta_p_i = P_i - P_{i-1}     consecutive diff
        Formula B (positional): delta_p_i = mu_p - P_i        global mean deviation

    The ordering computation is the ONLY thing that differs between A and B.
    Everything else — normaliser p_bar, components, output sum — is unchanged.

    Key distinction from Formula-B-sem
    ------------------------------------
        Formula B sem:  delta computed from V = TokenEmbedding(x)  (semantic space)
        Formula B pos:  delta computed from P = LegendrePositionEmbedding(x) (positional space)

    V does NOT participate in the computation of delta_p here.

    Tensor shapes
    -------------
        val       [B, L, D]   V_i  = TokenEmbedding(x)
        temp      [B, L, D]   T_i  = temporal_embedding(x_mark)
        leg       [B, L, D]   P_i  = LegendrePositionEmbedding(x)  (buffer, no grad)
        leg_d     [B, L, D]   leg.detach()
        mu_p      [B, 1, D]   global mean of P over sequence dim  → broadcasts
        delta_p   [B, L, D]   mu_p - leg_d
        p_bar     [B, 1, 1]   mean of per-token L2 norms of P     scalar per sample
        ordering  [B, L, D]   delta_p / (p_bar + 1e-8)
        output    [B, L, D]   dropout(val + temp + leg + ordering)
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,
                 max_len=5000):
        super(DataEmbedding_formula_b_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = (TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                                   if embed_type != 'timeF'
                                   else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq))
        import os, sys
        sys.path.insert(0, os.path.dirname(__file__))
        from legendre_embedding import LegendrePositionEmbedding
        self.legendre_embedding = LegendrePositionEmbedding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

        # Diagnostic step counter — prints signal statistics every 100 steps
        self._diag_step = 0

    def forward(self, x, x_mark):
        """
        Args:
            x      : [B, L, c_in]   raw input features
            x_mark : [B, L, d_time] time-stamp features

        Returns:
            [B, L, d_model]  embedded sequence
        """
        # ── standard components ───────────────────────────────────────────────
        val  = self.value_embedding(x)          # [B, L, D]   V_i
        temp = self.temporal_embedding(x_mark)  # [B, L, D]   T_i
        leg  = self.legendre_embedding(x)       # [B, L, D]   P_i  (buffer, no grad)

        # ── Formula B: global mean deviation in positional space ──────────────
        # Detach: leg is a fixed non-trainable buffer; no gradient needed
        leg_d = leg.detach()                                          # [B, L, D]

        # mu_p = global mean of P over the sequence dimension
        mu_p    = leg_d.mean(dim=1, keepdim=True)                    # [B, 1, D]  → broadcasts

        # delta_p_i = mu_p - P_i  (deviation of each positional vector from global mean)
        delta_p = mu_p - leg_d                                        # [B, L, D]

        # ── scalar normalisation over positional norms (unchanged from Formula A) ──
        # leg_d.norm(dim=-1): [B, L]
        # .mean(dim=1, keepdim=True): [B, 1]
        # .unsqueeze(-1): [B, 1, 1]
        p_bar = leg_d.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]

        ordering = delta_p / (p_bar + 1e-8)                          # [B, L, D]

        # ── diagnostics (every 100 training steps) ───────────────────────────
        if self.training and self._diag_step % 100 == 0:
            with torch.no_grad():
                val_norm      = val.norm(dim=-1).mean().item()
                ordering_norm = ordering.norm(dim=-1).mean().item()
                temp_norm     = temp.norm(dim=-1).mean().item()
                p_bar_mean    = p_bar.mean().item()
                print(f"[FORMULA_B_POS step={self._diag_step}] "
                      f"val_norm={val_norm:.4f} | "
                      f"ordering_norm={ordering_norm:.4f} | "
                      f"temp_norm={temp_norm:.4f} | "
                      f"p_bar={p_bar_mean:.4f} | "
                      f"ordering/val={ordering_norm / (val_norm + 1e-8):.4f}")
        if self.training:
            self._diag_step += 1

        # ── combine: V_i + T_i + P_i + Ordering_i ────────────────────────────
        return self.dropout(val + temp + leg + ordering)              # [B, L, D]
