# Color Latent Lab

Standalone experiments for testing how a language model internally represents color.

The repo now has one primary workflow and a set of older support workflows.

Primary workflow:

- `sae-geometry`: sweep the published Qwen SAEs over controlled `Color: X` prompts
- `sae-intervene`: inject discovered color directions into blank or semantic prompts

Support workflows:

- `sae-train` / `sae-analyze`: train and analyze a custom SAE

Legacy workflows:

- `run`
- `patch`
- `logit-lens`
- `summarize-logit-lens`
- `color-word-basis`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Primary workflow

### Run the off-the-shelf SAE geometry sweep

This is the cleaner `Color: X` experiment for mapping color-vs-format features using the published
Qwen SAEs.

```bash
color-latent-lab sae-geometry \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_sae_geometry \
  --sae-repo-id-or-path andyrdt/saes-qwen2.5-7b-instruct \
  --sae-layers 3,7,11,15,19,23,27 \
  --prompt-template 'Color: {value}'
```

By default this builds a mixed panel of:

- anchor prompts like `Color: red`, `Color: #ff0000`, `Color: 255,0,0`
- the bundled color-word catalog from `word_lists/english_color_words.txt`

It writes per-layer artifacts like:

- `layer_XX/encoded_pca.svg`
- `layer_XX/feature_scores.jsonl`
- `layer_XX/top_invariant_features.json`
- `layer_XX/family_feature_rankings.json`
- `layer_XX/directions/`
- `layer_summary.jsonl`

`sae-geometry` now checkpoints activation batches and completed layer analyses. If a run is
interrupted, rerun the same command with `--resume`.

If you want the faster path and do not care about silhouette metrics, add `--skip-silhouette`.

### Run a direction intervention

After a geometry run, inject one discovered family direction into blank or semantic prompts.

```bash
color-latent-lab sae-intervene \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --geometry-dir ./analysis/qwen_sae_geometry \
  --output-dir ./analysis/qwen_sae_intervene_red_l15 \
  --layer 15 \
  --family red \
  --prompt-mode blank_hex \
  --alpha-values -8,-4,-2,-1,0,1,2,4,8
```

For semantic prompts instead of `Hex code for:`:

```bash
color-latent-lab sae-intervene \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --geometry-dir ./analysis/qwen_sae_geometry \
  --output-dir ./analysis/qwen_sae_intervene_red_semantic_l15 \
  --layer 15 \
  --family red \
  --prompt-mode semantic_hex
```

`sae-intervene` also checkpoints completed prompt batches, so you can restart it with `--resume`
without redoing finished batches.

The geometry sweep plus interventions are the main recommended workflow.

## Support workflows

### Train a custom SAE

```bash
color-latent-lab sae-train \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_sae_word_anchors_layer16 \
  --layer 16 \
  --prompt-format word \
  --word-preset color_words \
  --activation-batch-size 32 \
  --train-batch-size 256 \
  --dictionary-multiplier 8 \
  --top-k 32 \
  --epochs 8
```

### Analyze a trained SAE

```bash
color-latent-lab sae-analyze \
  --sae-checkpoint-path ./analysis/qwen_sae_layer16/sae_checkpoint.pt \
  --color-run-dir ./analysis/qwen_cross_format \
  --output-dir ./analysis/qwen_sae_layer16_analysis \
  --layer 16 \
  --format-name all
```

## Legacy workflows

These older commands are still available, but they are no longer the recommended path for the
color-basis question.

### Cross-format residual sweep

```bash
color-latent-lab run \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_cross_format \
  --limit 1000 \
  --batch-size 16 \
  --grid-stride 4
```

This writes two probe tracks:

- `within_schema_probe_accuracy.jsonl`
- `cross_format_probe_transfer.jsonl`

### Residual patching

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

### Orchestrated color-word basis run

```bash
color-latent-lab color-word-basis \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_color_word_basis \
  --sae-layer 16
```

### Logit lens

```bash
color-latent-lab logit-lens \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_logit_lens \
  --limit 1000 \
  --batch-size 16 \
  --top-token-count 8
```

This now writes `interpretation.json` and `interpretation.md`, including statements like when semantic color appears and when hex or rgb formatting overtakes color-word tokens.

### Re-interpret an existing logit lens run

```bash
color-latent-lab summarize-logit-lens \
  --run-dir ./analysis/qwen_logit_lens
```

If a run is interrupted:

```bash
color-latent-lab run \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_cross_format \
  --limit 1000 \
  --batch-size 16 \
  --resume
```

For `--limit` values above the built-in noun seed list, the runner automatically pulls from a
system dictionary such as `/usr/share/dict/words`. If the machine does not have one, pass
`--word-list-path` explicitly instead of silently falling back to a tiny list.

You can also use `--word-preset color_words` to run on explicit color anchors instead of nouns.
For a larger hand-curated lexicon of English color terms, point `--word-list-path` at
`./word_lists/english_color_words.txt`.

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

- `layer_XX/encoded_pca.svg`
- `layer_XX/top_invariant_features.json`
- `layer_XX/family_feature_rankings.json`
- `layer_XX/directions/`
- `intervention_rows.jsonl`
- `alpha_summary.jsonl`
- `shared_pca_grid.svg`
- `format_transfer_curve.svg`
- `within_schema_probe_accuracy.jsonl`
- `cross_format_probe_transfer.jsonl`
- `probe_transfer.jsonl`
- `logit_lens_curve.svg`
- `layer_summary.jsonl`
- `top_token_snapshots.jsonl`
- `interpretation.json`
- `interpretation.md`
- `patched_predictions.jsonl`
- `report.md`
