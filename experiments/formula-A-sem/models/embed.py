import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """Standard sinusoidal positional embedding (Vaswani et al., 2017).
    Retained here for the vanilla baseline pe_mode only.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """1-D circular Conv over 7 raw input features → d_model dimensions.
    Identical to the original Informer (Zhou et al., 2021).
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """Sinusoidally-initialised, frozen embedding for calendar integers."""
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Fixed sinusoidal embeddings for hour, weekday, day, month integers."""
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

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
    """Linear projection over continuous time features (used with embed='timeF')."""
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla baseline embedding (unchanged from Zhou et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────

class DataEmbedding(nn.Module):
    """Zhou's original DataEmbedding — unchanged.

    Formula:
        X'_i = X_i + PE_i + T_i

    where:
        X_i   = TokenEmbedding(x)          value / semantic content
        PE_i  = sinusoidal positional       absolute position
        T_i   = TemporalEmbedding(x_mark)  calendar context
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != 'timeF'
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = (self.value_embedding(x)
             + self.position_embedding(x)
             + self.temporal_embedding(x_mark))
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4c — Delta Ordering in Positional Space
# ─────────────────────────────────────────────────────────────────────────────

class DataEmbedding_delta_pos(nn.Module):
    """Experiment 4c: Delta Ordering injected into the Positional Pathway.

    Research hypothesis
    -------------------
    Replace the sinusoidal positional embedding PE_i with a consecutive-delta
    ordering signal Δ(X_i) built directly from the token (value) embeddings.
    The signal lives in the *positional pathway* — it occupies exactly the slot
    that PE_i held in the baseline, while the semantic content X_i is added
    separately and is never modified.

    Embedding formula
    -----------------
        Baseline:  X'_i = X_i + PE_i + T_i

        Exp 4c:    X'_i = X_i + Δ(X_i) + T_i

    where the consecutive delta is:

        Δ(X_i) = X_i − X_{i−1}     for i ≥ 1
        Δ(X_1) = X_1 − 0           for i = 0  (zero-pad boundary)

    X_i here refers to TokenEmbedding(x) — the d_model-dimensional semantic
    representation produced by the 1-D circular convolution.

    Component inventory
    -------------------
        value embedding (X_i)              YES — unchanged
        temporal embedding (T_i)           YES — unchanged
        delta ordering signal (Δ(X_i))     YES — replaces PE_i
        sinusoidal PE                       NO  — removed
        Legendre embedding                  NO  — not used
        distance operator                   NO  — not used
        normalization                       NO  — raw delta (no division)

    Mathematical properties
    -----------------------
    * Translation-invariant: adding a constant c to all X_i leaves Δ(X_i)
      unchanged for i ≥ 1 (boundary term Δ(X_0) = X_0 shifts by c, which is
      a known, bounded artefact).
    * Order-sensitive: permuting the sequence changes Δ(X_i) at every affected
      position.  The signal encodes *direction of change* between consecutive
      token embeddings.
    * No new parameters: the delta is computed from existing TokenEmbedding
      outputs; no additional weight matrices are introduced.
    * Boundary condition: Δ(X_0) = X_0 − 0 = X_0, i.e., the first position
      receives its own embedding as its ordering signal.

    Relationship to baseline
    ------------------------
    The only structural change is:

        PE_i  →  Δ(X_i) = X_i − X_{i−1}

    All other components (TokenEmbedding, TemporalEmbedding, Encoder, Decoder,
    Attention, projection) are bit-for-bit identical to the baseline.
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_delta_pos, self).__init__()

        # Semantic content — unchanged from baseline
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # Temporal context — unchanged from baseline
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != 'timeF'
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )

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
        # ── semantic content ────────────────────────────────────────────────
        val = self.value_embedding(x)           # [B, L, D]   X_i

        # ── temporal context ────────────────────────────────────────────────
        temp = self.temporal_embedding(x_mark)  # [B, L, D]   T_i

        # ── delta ordering signal (replaces PE_i) ───────────────────────────
        # Δ(X_0) = X_0 − 0  →  delta[:, 0, :] = val[:, 0, :]  (already set)
        # Δ(X_i) = X_i − X_{i−1}  for i ≥ 1
        delta = torch.zeros_like(val)                        # [B, L, D]
        delta[:, 0, :] = val[:, 0, :]                       # boundary: Δ(X_1) = X_1
        delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]   # consecutive diff

        # ── diagnostics (every 100 training steps) ──────────────────────────
        if self.training and self._diag_step % 100 == 0:
            with torch.no_grad():
                val_norm   = val.norm(dim=-1).mean().item()
                delta_norm = delta.norm(dim=-1).mean().item()
                temp_norm  = temp.norm(dim=-1).mean().item()
                print(f"[DELTA_POS step={self._diag_step}] "
                      f"val_norm={val_norm:.4f} | "
                      f"delta_norm={delta_norm:.4f} | "
                      f"temp_norm={temp_norm:.4f} | "
                      f"delta/val={delta_norm / (val_norm + 1e-8):.4f}")
        if self.training:
            self._diag_step += 1

        # ── combine: X_i + Δ(X_i) + T_i ────────────────────────────────────
        return self.dropout(val + delta + temp)              # [B, L, D]
