# Color Latent Lab

Standalone experiments for testing whether a language model represents semantic color concepts
before it renders them into a requested output format like a color word, a hex code, or an RGB
triplet.

## What it does

- Runs the same prompt across `word`, `hex`, and `rgb` output formats
- Caches residual-stream states at every layer
- Computes shared PCA views colored by consensus color family
- Trains simple cross-format probes to test whether the representation transfers
- Patches residual states from one format into another to test causal transfer
- Trains a custom SAE on a chosen layer and analyzes which features separate color families
- Runs a logit-lens pass to measure when color-word tokens give way to hex or rgb formatting tokens
- Writes live heartbeat files for long Linux-box runs
- Checkpoints every batch and supports `--resume`
- Exports a compact results bundle for downstream review

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run the cross-format experiment

```bash
color-latent-lab run \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_cross_format \
  --limit 200 \
  --batch-size 16 \
  --grid-stride 4
```

## Run a causal patch experiment

```bash
printf 'fire,ocean\nrose,sky\nsun,night\n' > ./analysis/color_pairs.csv

color-latent-lab patch \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --run-dir ./analysis/qwen_cross_format \
  --output-dir ./analysis/qwen_patch_l16 \
  --source-format word \
  --target-format hex \
  --layer 16 \
  --pairs-path ./analysis/color_pairs.csv \
  --batch-size 4
```

## Train a custom SAE

```bash
color-latent-lab sae-train \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_sae_layer16 \
  --layer 16 \
  --prompt-format hex \
  --limit 10000 \
  --activation-batch-size 32 \
  --train-batch-size 256 \
  --dictionary-multiplier 8 \
  --top-k 32 \
  --epochs 8
```

## Run the logit lens pass

```bash
color-latent-lab logit-lens \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_logit_lens \
  --limit 200 \
  --batch-size 16 \
  --top-token-count 8
```

This now writes `interpretation.json` and `interpretation.md`, including statements like when semantic color appears and when hex or rgb formatting overtakes color-word tokens.

## Re-interpret an existing logit lens run

```bash
color-latent-lab summarize-logit-lens \
  --run-dir ./analysis/qwen_logit_lens
```

## Analyze a trained SAE

```bash
color-latent-lab sae-analyze \
  --sae-checkpoint-path ./analysis/qwen_sae_layer16/sae_checkpoint.pt \
  --color-run-dir ./analysis/qwen_cross_format \
  --output-dir ./analysis/qwen_sae_layer16_analysis \
  --layer 16 \
  --format-name all
```

If a run is interrupted:

```bash
color-latent-lab run \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_cross_format \
  --limit 200 \
  --batch-size 16 \
  --resume
```

## Heartbeats

Every run writes:

- `manifest.json`
- `heartbeat_status.json`
- `heartbeat_events.jsonl`
- `checkpoints/`
- `final_results.json`

Useful monitoring commands:

```bash
tail -f ./analysis/qwen_cross_format/heartbeat_events.jsonl
watch -n 5 'cat ./analysis/qwen_cross_format/heartbeat_status.json'
```

## Export

```bash
color-latent-lab export \
  --run-dir ./analysis/qwen_cross_format \
  --patch-dir ./analysis/qwen_patch_l16 \
  --logit-lens-dir ./analysis/qwen_logit_lens \
  --output-dir ./analysis/export_bundle
```

This copies the key files and writes `final_results_bundle.json`.

## Key artifacts

- `shared_pca_grid.svg`
- `format_transfer_curve.svg`
- `probe_transfer.jsonl`
- `logit_lens_curve.svg`
- `layer_summary.jsonl`
- `top_token_snapshots.jsonl`
- `interpretation.json`
- `interpretation.md`
- `patched_predictions.jsonl`
- `report.md`
