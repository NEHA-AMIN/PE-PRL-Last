# Experiment 3: Label Only (Legendre Polynomials)

## Objective
Test whether Legendre polynomial position embeddings (Label component) alone can provide meaningful positional information for time series forecasting.

## Hypothesis
Pure distinctiveness (orthogonal labeling) without ordering or distance decay.

## Implementation

### Mathematical Formulation
```
X'_i = X_i + P_i

where:
  X_i = value_embedding(x)      [Semantic content]
  P_i = Legendre(i)             [LABEL ONLY - Equation 1]
  
  NO temporal embedding
  NO distance operator
  NO ordering signal
```

### Label Component (Equation 1)
```
Legendre Polynomials: P_i = [L_0(x_i), L_1(x_i), ..., L_{d-1}(x_i)]

Orthogonality: ⟨P_n, P_m⟩ = { 1 if n=m, 0 if n≠m }
```

- Generated using scipy.special.legendre
- Positions normalized to [-1, 1]
- Scaled by 1/√d_model
- Pre-computed and cached

## Key Differences from Other Experiments

| Aspect | Exp 1 (D) | Exp 2 (L+O+D) | Exp 3 (L) |
|--------|-----------|---------------|-----------|
| **Label (L)** | ❌ | ✅ | ✅ ONLY |
| **Order (O)** | ❌ | ✅ | ❌ |
| **Distance (D)** | ✅ | ✅ | ❌ |
| **Temporal** | ✅ | ✅ | ❌ REMOVED |
| **Components** | α(i,j) bias | Full LOD | Pure Label |

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer |
| Attention | Full |
| Sequence Length | 96 |
| Prediction Length | 24 |
| Encoder Layers | 2 |
| Decoder Layers | 1 |
| d_model | 512 |
| Legendre Scaling | 1/√d_model |

## File Structure
```
experiments/exp3_label_only/
├── models/
│   ├── legendre_embedding.py   - Label component (from Exp2)
│   ├── embed.py                - MODIFIED: Label only
│   ├── attn.py                 - Vanilla (unchanged)
│   └── ... (other vanilla files)
├── README.md                    - This file
└── run_exp3.sh                  - Training script
```

## How to Run
```bash
bash experiments/exp3_label_only/run_exp3.sh
```

## Expected Output

Training for 6 epochs with early stopping.
Results saved to: `results/exp3_label_only/`

## Results

| Metric | Value |
|--------|-------|
| MSE    | - |
| MAE    | - |

*To be filled after running experiment*

## Comparison with Baselines

| Experiment | Components | MSE | MAE | Analysis |
|------------|-----------|-----|-----|----------|
| Vanilla | Standard PE | 0.519 | 0.513 | Baseline |
| Exp 1 | D | 0.725 | 0.652 | Distance alone |
| Exp 2 | L+O+D | 0.753 | 0.678 | Full LOD |
| **Exp 3** | **L** | **-** | **-** | **Label alone** |

## Analysis Questions

1. Does orthogonal distinctiveness provide any benefit?
2. Is Label component meaningful without Order/Distance?
3. How does pure Label compare to pure Distance (Exp1)?
4. Does removing temporal embedding hurt more than adding Label helps?

## Theoretical Justification

**Legendre Polynomials Provide:**
- ✅ Distinctiveness (orthogonality)
- ✅ Complete basis (spans function space)
- ❌ NO ordering information
- ❌ NO distance decay
- ❌ NO temporal semantics

**This tests:** Can position be encoded through distinctiveness alone?
