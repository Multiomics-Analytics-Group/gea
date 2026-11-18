"""Feature intervention: does activating SAE features change the classifier's prediction?

For graph-level embeddings. Takes misclassified samples, edits their embeddings
along the SAE decoder direction of features associated with their TRUE class
(intervention.feature_intervention), and checks whether the classifier's
prediction flips to the correct class. Controls:

  - random live features (same number of features, several random draws)
  - features of the WRONG class

Inputs from the GEA workflow (run it first, up to interpretation_plots.py):
  --embeddings_path                  the graph embeddings .npz used by all stages
  --splits_path, --checkpoint_path   from train_sae.py
  --sae_annotation_association_path  from sae_annotation_association.py
                                     (sample activations + training maxima)
  --feature_annotations_path         feature_annotations.pt, saved by
                                     interpretation_plots.py in its --output_dir

Features are either chosen automatically or given by hand with --features.
Automatic selection: every annotated feature is scored with Fisher's exact
test (active in correctly classified positive vs negative samples, as in
GEA's discriminative feature plot), p-values are Benjamini-Hochberg adjusted,
and the --top_k per class with adjusted p < --alpha are kept; each feature's
class is the class it is more often active in (odds ratio).

The classifier is the prediction head that takes these embeddings as input,
saved as TorchScript, e.g.:

    head = model.classifier.eval()
    torch.jit.trace(head, example_embeddings).save("classifier.pt")

It must output logits: (N, 2) for two classes, or (N,) / (N, 1) for the
positive class. Before any edit, the script checks that it reproduces the
predictions stored in the embeddings file.

Outputs (in --output_dir):
  intervention_results.csv   one row per (sample, feature set, strength, condition)
  intervention_summary.csv   flip rates per condition and strength
  selected_features.csv      all scored features, with p_adjusted and "selected"
  1_dose_response.png, 2_per_feature.png, 3_per_sample.png
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# Make the modules next to this script importable from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gea.gea import ShallowSAE  # noqa: E402
from intervention import (  # noqa: E402
    discriminative_feature_table,
    feature_intervention,
    random_feature_sets,
    select_intervention_features,
)
from intervention_plots import (  # noqa: E402
    INTERVENTION_CONDITIONS,
    plot_intervention_dose_response,
    plot_intervention_per_feature,
    plot_intervention_per_sample,
)


def parse_features(values):
    """'1427:1 702:0' -> {1: [1427], 0: [702]}"""
    selected = {}
    for item in values:
        feature, label = item.split(":")
        selected.setdefault(int(label), []).append(int(feature))
    return selected


def main(args):
    device = args.device if (
        args.device.startswith("cpu") or torch.cuda.is_available()
    ) else "cpu"

    class_names = (
        {int(k): v for k, v in args.class_names.items()}
        if args.class_names is not None
        else {0: "class_0", 1: "class_1"}
    )
    if len(class_names) != 2 or args.positive_label not in class_names:
        raise ValueError("class_names must contain exactly two classes, including positive_label.")
    labels = sorted(class_names)

    os.makedirs(args.output_dir, exist_ok=True)

    # Data, SAE, training maxima, classifier
    data = np.load(args.embeddings_path, allow_pickle=True)
    splits = torch.load(args.splits_path, map_location="cpu", weights_only=False)

    sae = ShallowSAE(
        in_dim=args.d_z,
        latent_dim=args.d_z * args.latent_multiplier,
        sparsity_weight=args.sparsity_weight,
    )
    sae.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu", weights_only=True))

    association = torch.load(args.sae_annotation_association_path, map_location="cpu", weights_only=False)
    maxima = torch.as_tensor(association["train_feature_maxima"]).float().reshape(-1)

    classifier = torch.jit.load(args.classifier_path, map_location=device).eval()

    # Feature selection
    table = None
    feature_labels = {}

    if args.feature_annotations_path is not None:
        feature_annotations = torch.load(args.feature_annotations_path, weights_only=False)
        table = discriminative_feature_table(
            association, feature_annotations,
            class_names=class_names,
            positive_label=args.positive_label,
            probability_threshold=args.probability_threshold,
            min_active=args.min_active,
        )
        feature_labels = dict(zip(table["feature_id"].astype(int), table["annotation"]))

    if args.features:
        selected = parse_features(args.features)
        for label in labels:
            selected.setdefault(label, [])
        print("Using features given with --features.")

    elif table is not None:
        selected, table = select_intervention_features(
            table, alpha=args.alpha, top_k=args.top_k, labels=labels,
        )
        table.to_csv(os.path.join(args.output_dir, "selected_features.csv"), index=False)

    else:
        raise ValueError("Give --feature_annotations_path (from interpretation_plots.py) or --features.")

    for label in labels:
        print(f"Features for '{class_names[label]}': {selected[label] or 'none'}")

    all_selected = [f for features in selected.values() for f in features]
    if not all_selected:
        raise ValueError("No features selected. Try a larger --alpha, or give --features.")

    common = dict(
        sae_model=sae,
        classifier=classifier,
        embeddings=data["embeddings"],
        targets=data["target"],
        feature_maxima=maxima,
        splits=splits,
        use_splits=tuple(args.use_splits),
        positive_label=args.positive_label,
        probability_threshold=args.probability_threshold,
        predictions=None if args.no_prediction_check else data["prediction"],
        prediction_atol=args.prediction_atol,
        sample_ids=data["entities"] if "entities" in data.files else None,
        device=device,
    )

    cond = INTERVENTION_CONDITIONS
    results = []

    def run(condition, features, target_class, **kwargs):
        df = feature_intervention(features=features, target_class=target_class, **common, **kwargs)
        if not df.empty:
            df["condition"] = condition
            df["class"] = class_names[target_class]
            results.append(df)

    for i, label in enumerate(labels):
        own = selected[label]
        other = [f for other_label in labels if other_label != label for f in selected[other_label]]
        if not own:
            print(f"No features for '{class_names[label]}'; skipping its misclassified samples.")
            continue

        activate = dict(samples="misclassified", strengths=tuple(args.strengths))

        # Main: true-class features, each alone and all together.
        run(cond["single"], own, label, **activate)
        run(cond["set"], [own], label, **activate)

        # Control: random live features (same numbers), excluding all selected features.
        run(cond["single_random"],
            random_feature_sets(maxima, 1, args.n_random, exclude=all_selected, seed=args.seed + i),
            label, **activate)
        run(cond["set_random"],
            random_feature_sets(maxima, len(own), args.n_random, exclude=all_selected,
                                seed=args.seed + 100 + i),
            label, **activate)

        # Control: features of the wrong class.
        if other:
            run(cond["set_wrong"], [other], label, **activate)

    if not results:
        print("No samples matched the selection (e.g. no misclassified samples in these splits).")
        return

    df = pd.concat(results, ignore_index=True)
    df["delta_toward_correct"] = np.where(
        df["target"] == args.positive_label, df["delta_prob"], -df["delta_prob"]
    )
    df.to_csv(os.path.join(args.output_dir, "intervention_results.csv"), index=False)

    summary = (
        df.groupby(["condition", "strength"])
        .agg(
            n_samples=("sample", "nunique"),
            flip_to_correct=("flipped_to_correct", "mean"),
            mean_delta_toward_correct=("delta_toward_correct", "mean"),
        )
        .round(3)
    )
    summary.to_csv(os.path.join(args.output_dir, "intervention_summary.csv"))
    print("\n" + summary.to_string())

    # Plots
    plot_intervention_dose_response(
        df, output_path=os.path.join(args.output_dir, "1_dose_response.png"))
    plot_intervention_per_feature(
        df, sort_strength=args.plot_strength, feature_labels=feature_labels,
        output_path=os.path.join(args.output_dir, "2_per_feature.png"))
    plot_intervention_per_sample(
        df, strength=args.plot_strength,
        output_path=os.path.join(args.output_dir, "3_per_sample.png"))

    print(f"\nResults and plots saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Activate SAE features in misclassified graph embeddings and test whether the prediction changes."
    )

    # Inputs from the GEA workflow
    parser.add_argument("--embeddings_path", required=True, help="Graph embeddings .npz (same file as the GEA stages).")
    parser.add_argument("--splits_path", required=True, help="Splits file from train_sae.py.")
    parser.add_argument("--checkpoint_path", required=True, help="Trained SAE weights from train_sae.py.")
    parser.add_argument("--sae_annotation_association_path", required=True,
                        help="Output of sae_annotation_association.py (activations and training maxima).")
    parser.add_argument("--feature_annotations_path", default=None,
                        help="feature_annotations.pt saved by interpretation_plots.py. Needed for "
                             "automatic feature selection and for feature names in the plots.")
    parser.add_argument("--d_z", type=int, default=1200, help="Embedding dimension.")
    parser.add_argument("--latent_multiplier", type=int, default=8)
    parser.add_argument("--sparsity_weight", type=float, default=1e-3)

    # Classifier
    parser.add_argument("--classifier_path", required=True,
                        help="TorchScript file of the prediction head (embeddings -> logits).")
    parser.add_argument("--no_prediction_check", action="store_true",
                        help="Skip the check that the classifier reproduces the stored predictions.")
    parser.add_argument("--prediction_atol", type=float, default=1e-4,
                        help="Allowed difference between recomputed and stored predictions (default: 1e-4).")

    # Feature selection
    parser.add_argument("--features", nargs="+", default=None, metavar="FEATURE:LABEL",
                        help="Manual features with the class they favour, e.g. 1427:1 702:0. "
                             "Overrides automatic selection.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Maximum FDR-adjusted Fisher p-value for automatic selection (default: 0.05).")
    parser.add_argument("--top_k", type=int, default=5, help="Features per class for automatic selection (default: 5).")
    parser.add_argument("--min_active", type=int, default=10,
                        help="Minimum number of correctly classified samples in which a feature is active "
                             "to be scored (default: 10, as in interpretation_plots.py).")

    # Intervention
    parser.add_argument("--use_splits", nargs="+", default=["test"], choices=["train", "val", "test"],
                        help="Splits to take samples from (default: test).")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1, 2],
                        help="Target activations, in units of each feature's training maximum (default: 0.5 1 2).")
    parser.add_argument("--n_random", type=int, default=20, help="Random feature draws per control (default: 20).")
    parser.add_argument("--seed", type=int, default=0)

    # Classes
    parser.add_argument("--class_names", type=json.loads, default=None,
                        help='JSON mapping of class labels to names, e.g. \'{"0": "class_0", "1": "class_1"}\'.')
    parser.add_argument("--positive_label", type=int, default=1)
    parser.add_argument("--probability_threshold", type=float, default=0.5)

    # Output
    parser.add_argument("--output_dir", default="intervention_results")
    parser.add_argument("--plot_strength", type=float, default=1.0,
                        help="Strength shown in the per-sample plot, and used to order the rows "
                             "of the per-feature plot (default: 1).")
    parser.add_argument("--device", default="cuda")

    main(parser.parse_args())
