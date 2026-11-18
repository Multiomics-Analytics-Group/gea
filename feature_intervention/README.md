# Feature intervention

Tests whether the SAE features found by GEA actually drive a classifier's
predictions. For misclassified samples, the graph embedding is edited along the
SAE decoder direction of features associated with the sample's **true** class,
and the classifier is run again. If the prediction flips to the correct class,
more often than with random features, the classifier relies on those features.

For each selected feature `f` with training maximum `max_f`, at a given strength:

```
z       = SAE encoder(x)                       current feature activations
z_new_f = max(z_f, strength × max_f)           never lowers a feature
x_new   = x + Σ_f (z_new_f − z_f) × W_dec[f]   only the chosen features change
```

Controls: random live SAE features (same number of features, several draws)
and the other class's selected features.

## Requirements

- **Graph-level embeddings** (one embedding per sample) and **binary classification**.
- The four GEA workflow scripts already run, in this order, on the same
  embeddings `.npz`: `train_sae.py`, `gea_pipeline.py`,
  `sae_annotation_association.py` and `interpretation_plots.py`. The
  intervention reads files they produce, so it is run after them.
- The classifier's **prediction head**: the module that takes these embeddings
  as input and outputs logits, saved as TorchScript:

  ```python
  head = model.classifier.eval()                 # your model's prediction head
  torch.jit.trace(head, example_embeddings).save("classifier.pt")
  ```

  It must output `(N, 2)` logits, or `(N,)` / `(N, 1)` for the positive class.
  The `prediction` field of the embeddings `.npz` must come from this head (the
  script checks this before editing anything; `--no_prediction_check` skips it).

## Usage

```bash
python feature_intervention/feature_intervention.py \
    --embeddings_path graph_embeddings.npz \
    --splits_path splits.pt \
    --checkpoint_path sae_model.pt \
    --sae_annotation_association_path sae_annotation_association.pt \
    --feature_annotations_path gea_results/feature_annotations.pt \
    --d_z 1200 --latent_multiplier 8 \
    --classifier_path classifier.pt \
    --class_names '{"0": "class_0", "1": "class_1"}' \
    --output_dir intervention_results
```

`--d_z`, `--latent_multiplier` and `--sparsity_weight` must match the values
used to train the SAE.

**Feature selection.** By default, every annotated feature is scored with
Fisher's exact test (active in correctly classified samples of one class vs the
other, as in GEA's discriminative feature plot). P-values are
Benjamini-Hochberg adjusted, and the `--top_k` features per class with adjusted
p < `--alpha` are kept. Each feature's class is the class it is more often
active in. Use `--features 1427:1 702:0` (feature:class) to choose by hand.

Other options: `--use_splits` (default `test`; e.g. `val test`), `--strengths`
(default `0.5 1 2`, in units of each feature's training maximum), `--n_random`
(random draws per control, default 20), `--min_active`, `--plot_strength`.
Run with `--help` for all options.

## Outputs

| File | Content |
|---|---|
| `intervention_results.csv` | One row per sample × feature set × strength × condition |
| `intervention_summary.csv` | Flip-to-correct rate and mean probability change per condition and strength |
| `selected_features.csv` | All scored features with adjusted p-values; `selected` marks the ones used |
| `1_dose_response.png` | All selected features of a class together vs random sets and the other class's features, per class |
| `2_per_feature.png` | Each selected feature on its own, one panel per strength, vs random single features of the same class |
| `3_per_sample.png` | Each misclassified sample's probability of its correct class: before, after random sets, after its true-class features |

## Files

- `feature_intervention.py`: the command-line script.
- `intervention.py`: feature scoring and selection, random controls, and the intervention itself.
- `intervention_plots.py`: the three plots.
