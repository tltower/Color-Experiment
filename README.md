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

## Code layout

The repo is now split so the large workflows do not all depend on one monolithic helper file:

- `src/color_latent_lab/color_formats.py`: prompt schemas, completion parsing, and family palettes
- `src/color_latent_lab/model_utils.py`: prompt rendering, device moves, last-token indexing, hook helpers
- `src/color_latent_lab/run_support.py`: heartbeat logging, checkpoint state, JSON/JSONL helpers
- `src/color_latent_lab/workflow_common.py`: shared stack loading, layer selection, word loading, mean helpers
- `src/color_latent_lab/format_analysis.py`: legacy cross-format PCA/probe analysis and SVG generation

The big workflow files still exist, but they now compose these smaller pieces instead of each
reimplementing them.

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

If you want exact color-word labels to repeat across schemas for held-out probe work, mirror the
catalog across formats:

```bash
color-latent-lab sae-geometry \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./analysis/qwen_sae_geometry_wordcodomain \
  --sae-repo-id-or-path andyrdt/saes-qwen2.5-7b-instruct \
  --sae-layers 11,15 \
  --catalog-formats word,hex,rgb \
  --prompt-template 'Color: {value}'
```

That writes `color_label` into `panel.jsonl`, using the same underlying color word across
`word`, `hex`, and `rgb` prompts when an approximate display color is available.

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

### Compare residual-stream probes vs SAE-code probes

This is a post-hoc analysis over an existing geometry run. It uses the same held-out splits for the
raw last-token residual activations in `activations/layer_XX.npy` and the SAE codes in
`layer_XX/encoded_features.npy`.

```bash
color-latent-lab probe-compare \
  --geometry-dir ./analysis/qwen_sae_geometry_wordcodomain \
  --output-dir ./analysis/qwen_sae_geometry_wordcodomain_probe_compare \
  --label-mode color_word \
  --schema-filter word,hex,rgb \
  --center-mode schema
```

Important: exact `color_word` probes only make sense when the same label appears more than once in
the selected panel. The mirrored `--catalog-formats word,hex,rgb` geometry run above is the
intended setup.

### Characterize discovered directions

After a focused geometry run, summarize what the saved family directions are doing:

```bash
python scripts/direction_characterization.py \
  --geometry-dir ./analysis/qwen_sae_geometry_wordcodomain \
  --output-dir ./analysis/qwen_sae_geometry_wordcodomain_direction_characterization \
  --layers 11,15
```

This writes per-layer artifacts such as:

- `direction_cosine_similarity.json` / `.svg`
- `direction_color_wheel.svg`
- `feature_attribution.json`
- `direction_feature_cards.md`
- `effective_dimensionality.json`
- `warm_cool_projection.json`
- `opponent_structure.json`

If you have already run interventions, pass `--intervention-root` to correlate causal steerability
with direction strength.

`direction_feature_cards.md` is the intended pre-intervention inspection step: it surfaces the top
features behind each family direction and the highest-activating prompts for those features.

### Build a broad intervention suite

If you want to sweep many directions, prompt templates, and far more than the 12 core hex/rgb
anchors, build a suite from the bundled color lexicon:

```bash
python scripts/direction_experiment_suite.py \
  --geometry-dir ./analysis/qwen_sae_geometry_wordcodomain \
  --output-dir ./analysis/qwen_direction_suite \
  --layers 11,15 \
  --catalog-limit 96
```

This writes:

- `prompt_files/` with broad catalog-driven `hex`, `rgb`, `word`, and description prompts
- `suite_manifest.json` with one entry per run
- `run_suite.sh` to launch the whole matrix

By default the suite uses the `focused` profile, which keeps the prompt sets relatively tight:

- generation prompts like `Hex code for: cerulean` and `RGB triplet for: cerulean`
- description prompts like `Describe the appearance of the color #007ba7 in three adjectives`

If you want the bigger exploratory matrix, add `--suite-profile extended`.

You can then launch the matrix directly by rerunning the command with `--run`.

### Analyze description-space outputs

For description-mode intervention runs, analyze shared descriptor words with cosine similarity and
linear probes:

```bash
python scripts/description_space_report.py \
  --intervention-root ./analysis/qwen_direction_suite/runs \
  --suite-manifest ./analysis/qwen_direction_suite/suite_manifest.json \
  --output-dir ./analysis/qwen_direction_suite_description_report
```

This writes per-run alpha cosine heatmaps plus `probe_summary.json`, which reports how linearly
predictable the steered direction and alpha-sign are from shared description words alone.

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
