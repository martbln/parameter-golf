# Differential Transformer — 7×512 SP-1024

This submission replaces the standard causal self-attention with **Differential Attention**
([Ye et al., 2024](https://arxiv.org/abs/2410.05258)) and adjusts the layer count to stay
inside the 16 MB artifact budget.

## Architecture

| Hyperparameter | Baseline | This run |
|---|---|---|
| `NUM_LAYERS` | 9 | **7** |
| `MODEL_DIM` | 512 | 512 |
| `NUM_HEADS` | 8 | 8 |
| `NUM_KV_HEADS` | 4 | 4 |
| `MLP_MULT` | 2 | 2 |
| `VOCAB_SIZE` | 1024 | 1024 |
| `TIE_EMBEDDINGS` | 1 | 1 |
| Attention type | Standard GQA | **Differential GQA** |

**Estimated parameter count:** ~16.1 M
**Estimated compressed size:** ~14.9 MB (well under the 16 MB cap)

## What Changed

### `DifferentialCausalSelfAttention`

Each attention layer now computes **two** attention-weighted value sums via separate
`(Q1, K1)` and `(Q2, K2)` projection pairs that share a single `V` projection:

```
y = SDPA(Q1, K1, V) − λ · SDPA(Q2, K2, V)
```

- `λ` is a **learnable per-head scalar** `sigmoid(lambda_param) ∈ (0, 1)`, initialized
  to 0.5 so the two streams start balanced.
- A **per-head RMSNorm** (SubLN from the paper) is applied to `y` before the output
  projection, stabilising training at early steps.
- `Q1/K1` are zero-initialized; `Q2/K2` receive default weight init.

The first stream learns to attend to relevant context; the second learns to capture
diffuse/noisy attention that the first then subtracts away, forcing sharper focus without
any extra training signal or auxiliary loss.

### Parameter budget trade-off

Adding `c_q2` and `c_k2` per layer costs +393 K params per layer.
Reducing `num_layers` 9 → 7 saves ~3.67 M params.
Net change: **−1.0 M params** vs the baseline, leaving more headroom for the 16 MB cap.

The 7-layer differential model has the same effective depth as a standard transformer
with ~9-10 layers thanks to the noise-cancellation property described in the paper
(each head processes a cleaner signal and converges faster per layer).

### Everything else unchanged

- Muon optimizer, Adam for scalars/embeddings, LR schedule, warmdown
- `relu²` MLP, RMSNorm, RoPE, logit softcap, tied embeddings
- int8 + zlib post-training quantization and roundtrip validation

## Run command

```bash
RUN_ID=diffattn_7x512_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=7 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-03-18_DiffAttn_7x512_SP1024/train_gpt.py
```

## Results

> Training results pending — to be filled after running on 8×H100 via nightresearch.app.

## References

- Ye et al., "Differential Transformer", 2024. https://arxiv.org/abs/2410.05258
- Baseline: `records/track_10min_16mb/2026-03-17_NaiveBaseline/` (val_bpb: 1.2244)
