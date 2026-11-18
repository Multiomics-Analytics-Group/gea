"""Feature intervention: the analysis functions.

Edits graph embeddings along SAE decoder directions and records how a
classifier's prediction changes. Self-contained: needs only numpy, pandas,
torch, scipy and statsmodels (plus a trained SAE from the GEA workflow).
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ── Feature intervention ───────────────────────────────────────────────────────

def _positive_probability(classifier, x, positive_label=1):
    """Probability of `positive_label` from a classifier returning logits.

    Two or more outputs are treated as class logits (softmax); a single output
    as the logit of the positive class (sigmoid).
    """
    logits = classifier(x)

    if logits.ndim == 1 or logits.shape[1] == 1:
        return torch.sigmoid(logits.reshape(-1))

    return torch.softmax(logits, dim=1)[:, positive_label]


def _as_feature_sets(features):
    """Normalise `features` to {set name: [feature ids]}."""
    if isinstance(features, dict):
        return {str(name): [int(f) for f in fs] for name, fs in features.items()}

    if isinstance(features, (int, np.integer)):
        return {str(int(features)): [int(features)]}

    features = list(features)

    if all(isinstance(f, (int, np.integer)) for f in features):
        # A flat list: each feature on its own.
        return {str(int(f)): [int(f)] for f in features}

    # A list of lists: each inner list is one set, edited together.
    return {"+".join(str(int(f)) for f in fs): [int(f) for f in fs] for fs in features}


def discriminative_feature_table(
    class_activations,
    feature_annotations,
    class_names=None,
    positive_label=1,
    probability_threshold=0.5,
    min_active=10,
    accepted_only=True,
    threshold_key="threshold",
):
    """Score every annotated SAE feature for how well it separates the classes.

    Uses the same steps and inputs as gea.results.plot_discriminative_feature_proportions,
    but returns ALL scored features (not only the plotted top features):

      1. For each feature, take its best eligible categorical annotation and
         that annotation's activation threshold.
      2. Normalise sample activations by the training feature maxima.
      3. Split samples into correctly / incorrectly classified, per class.
      4. For each feature active in at least `min_active` correctly classified
         samples: Fisher's exact test of "active above threshold" in correctly
         classified positive vs negative samples.

    Parameters
    ----------
    class_activations : dict
        Output of scripts/gea/sae_annotation_association.py (needs
        "train_feature_maxima" and "results").
    feature_annotations : dict
        Output of gea.results.build_feature_annotations, saved by
        scripts/gea/interpretation_plots.py as feature_annotations.pt.

    Returns
    -------
    pd.DataFrame
        One row per scored feature, ranked by Fisher p-value: feature_id,
        annotation, threshold, n_active, frequency_difference, odds_ratio,
        fisher_p_value, favoured_label, favoured_class (odds ratio > 1: more
        often active in correctly classified positive samples).
    """
    if class_names is None:
        class_names = {0: "class_0", 1: "class_1"}

    if len(class_names) != 2 or positive_label not in class_names:
        raise ValueError(
            "class_names must contain exactly two classes, including positive_label."
        )

    negative_label = next(label for label in class_names if label != positive_label)
    positive_name = class_names[positive_label]
    negative_name = class_names[negative_label]

    # 1. Best eligible annotation (and its threshold) per feature.
    best_annotations = {}

    for feature_id, annotation_groups in feature_annotations.items():
        candidates = sorted(
            annotation_groups.get("categorical", []),
            key=lambda ann: ann.get("rank", float("inf")),
        )
        candidates = [
            ann for ann in candidates
            if (not accepted_only or ann.get("accepted", False))
            and ann.get(threshold_key) is not None
        ]
        if candidates:
            best_annotations[int(feature_id)] = {
                "annotation": candidates[0]["annotation"],
                "threshold": float(candidates[0][threshold_key]),
            }

    if not best_annotations:
        raise ValueError("No eligible categorical annotations were found.")

    # 2. Sample activations, normalised by the training maxima.
    train_feature_maxima = torch.as_tensor(
        class_activations["train_feature_maxima"]
    ).float().cpu().numpy().reshape(-1)

    activation_rows, targets, predictions = [], [], []
    for record in class_activations["results"].values():
        activation = torch.as_tensor(record["activations"]).float().cpu().numpy().reshape(-1)
        if len(activation) != len(train_feature_maxima):
            raise ValueError("Activation vector and training maxima have different lengths.")
        activation_rows.append(activation)
        targets.append(int(record["target"]))
        predictions.append(float(record["prediction"]))

    if not activation_rows:
        raise ValueError("No sample activations were found.")

    activations = np.asarray(activation_rows)
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values.")
    if not np.isin(targets, list(class_names)).all():
        raise ValueError("Targets contain labels not present in class_names.")

    valid_maxima = np.isfinite(train_feature_maxima) & (train_feature_maxima > 0)
    normalized = np.full_like(activations, np.nan, dtype=float)
    normalized[:, valid_maxima] = activations[:, valid_maxima] / train_feature_maxima[valid_maxima]

    # 3. Correctly / incorrectly classified samples, per class.
    predicted_labels = np.where(predictions >= probability_threshold, positive_label, negative_label)
    correct = targets == predicted_labels

    group_masks = {
        f"correct_{positive_name}": correct & (targets == positive_label),
        f"misclassified_{positive_name}": ~correct & (targets == positive_label),
        f"correct_{negative_name}": correct & (targets == negative_label),
        f"misclassified_{negative_name}": ~correct & (targets == negative_label),
    }
    if not group_masks[f"correct_{positive_name}"].any() or not group_masks[f"correct_{negative_name}"].any():
        raise ValueError("Both classes must contain correctly classified samples.")

    group_activations = {name: normalized[mask] for name, mask in group_masks.items()}

    # 4. Fisher's exact test per annotated feature.
    features = np.array(sorted(best_annotations), dtype=int)
    features = features[(features >= 0) & (features < len(train_feature_maxima))]
    features = features[valid_maxima[features]]

    rows = []
    for feature_id in features:
        threshold = best_annotations[feature_id]["threshold"]

        group_values = {
            name: values[:, feature_id][np.isfinite(values[:, feature_id])]
            for name, values in group_activations.items()
        }
        pos = group_values[f"correct_{positive_name}"]
        neg = group_values[f"correct_{negative_name}"]
        if len(pos) == 0 or len(neg) == 0:
            continue

        pos_active = int((pos > threshold).sum())
        neg_active = int((neg > threshold).sum())
        n_active = pos_active + neg_active
        if n_active < min_active:
            continue

        odds_ratio, p_value = fisher_exact(
            [[pos_active, len(pos) - pos_active], [neg_active, len(neg) - neg_active]],
            alternative="two-sided",
        )

        rows.append({
            "feature_id": int(feature_id),
            "annotation": best_annotations[feature_id]["annotation"],
            "threshold": threshold,
            "n_active": n_active,
            "frequency_difference": pos_active / len(pos) - neg_active / len(neg),
            "odds_ratio": odds_ratio,
            "fisher_p_value": p_value,
            "favoured_label": positive_label if odds_ratio > 1 else negative_label,
            "favoured_class": positive_name if odds_ratio > 1 else negative_name,
        })

    if not rows:
        raise ValueError("No valid categorical annotations met the selection criteria.")

    table = pd.DataFrame(rows)
    return table.sort_values(
        ["fisher_p_value", "frequency_difference"],
        key=lambda col: col if col.name == "fisher_p_value" else -col.abs(),
    ).reset_index(drop=True)


def select_intervention_features(
    table,
    alpha=0.05,
    top_k=5,
    labels=(0, 1),
):
    """Pick features for intervention from the discriminative feature table.

    The table is the output of discriminative_feature_table (or its saved
    CSV): one row per scored feature with its Fisher p-value, odds ratio and
    favoured class. Steps:

      1. Benjamini-Hochberg FDR correction over all features in the table.
      2. Keep features with adjusted p < alpha.
      3. Per class (favoured_label), keep the top_k by p-value.

    Parameters
    ----------
    table : pd.DataFrame or str
        The table, or a path to its CSV.

    Returns
    -------
    selected : dict
        {class label: [feature ids]}; a class may have an empty list.
    table : pd.DataFrame
        The input table with "p_adjusted" and "selected" columns added.
    """
    if not isinstance(table, pd.DataFrame):
        table = pd.read_csv(table)

    table = table.copy()
    p_values = table["fisher_p_value"].fillna(1.0).to_numpy()
    table["p_adjusted"] = multipletests(p_values, method="fdr_bh")[1]

    significant = table[table["p_adjusted"] < alpha]
    selected = {}

    for label in labels:
        rows = significant[significant["favoured_label"] == label]
        rows = rows.sort_values("fisher_p_value").head(top_k)
        selected[int(label)] = rows["feature_id"].astype(int).tolist()

    chosen = {f for features in selected.values() for f in features}
    table["selected"] = table["feature_id"].isin(chosen)

    return selected, table


def random_feature_sets(
    feature_maxima,
    set_size,
    n_sets=20,
    exclude=(),
    seed=0,
):
    """Draw random sets of live SAE features, as a control for feature_intervention.

    Only features that were ever active in training (maximum > ~0) are drawn,
    since editing a dead feature direction has no counterpart in real data.

    Returns
    -------
    dict
        {"random_<i>": [feature ids]}, ready to pass as `features`.
    """
    maxima = torch.as_tensor(feature_maxima).float().reshape(-1).cpu().numpy()
    live = np.flatnonzero(maxima > 1e-6)
    live = np.setdiff1d(live, np.asarray(list(exclude), dtype=int))

    if len(live) < set_size:
        raise ValueError(
            f"Only {len(live)} live features available; cannot draw sets of {set_size}."
        )

    rng = np.random.default_rng(seed)

    return {
        f"random_{i}": sorted(rng.choice(live, size=set_size, replace=False).tolist())
        for i in range(n_sets)
    }


def feature_intervention(
    sae_model,
    classifier,
    embeddings,
    targets,
    features,
    feature_maxima,
    splits=None,
    use_splits=("test",),
    samples="misclassified",
    target_class=None,
    strengths=(0.5, 1, 2),
    predictions=None,
    positive_label=1,
    probability_threshold=0.5,
    sample_ids=None,
    device="cpu",
    prediction_atol=1e-4,
):
    """Edit SAE features in embeddings and record how the classifier's prediction changes.

    For each selected sample, feature set and strength, the embedding is edited
    along the SAE decoder direction of each feature:

        z        = SAE.encode(x)                       current feature activations
        z_new_f  = max(z_f, strength * max_f)          (never lowers a feature)
        x_new    = x + sum_f (z_new_f - z_f) * W_dec[f]

    Only the chosen features change; the part of x the SAE does not reconstruct
    is left untouched. The classifier is then applied to x and x_new.

    Parameters
    ----------
    sae_model : ShallowSAE
        Trained SAE on the same embeddings.
    classifier : callable
        Prediction head: embeddings (N, D) -> logits (N, C) or (N,) / (N, 1).
        Use the model in eval mode.
    embeddings : array or tensor (N, D)
        All embeddings (e.g. the full dataset); selection is done here.
    targets : array (N,)
        True class labels (0/1).
    features : int, list of int, list of lists, or dict
        A flat list edits each feature on its own; a list of lists (or a dict
        {name: [ids]}) edits each inner set together.
    feature_maxima : array (n_latent,)
        Per-feature maximum activation on the training split. Strengths are in
        these units: 1 = as strong as the feature ever was in training.
    splits : dict or None
        {"train": idx, "val": idx, "test": idx}, e.g. the splits file used to
        train the SAE. None uses all rows.
    use_splits : tuple of str
        Which splits to take samples from, e.g. ("test",) or ("val", "test").
    samples : {"misclassified", "correct", "all"}
        Which samples to edit, judged by the classifier on the unedited embeddings.
    target_class : int or None
        Keep only samples whose true class is this label (e.g. activate
        soluble-associated features only in truly soluble molecules).
    strengths : tuple of float
        Target activations, in units of each feature's training maximum.
    predictions : array (N,) or None
        Stored probabilities of `positive_label`. If given, the classifier must
        reproduce them on the unedited embeddings (within `prediction_atol`),
        otherwise a ValueError is raised: the head and embeddings do not match.
    sample_ids : array (N,) or None
        Identifiers carried into the output (e.g. SMILES). Defaults to row numbers.

    Returns
    -------
    pd.DataFrame
        One row per (sample, feature set, strength) with the original and new
        probability of `positive_label`, the change, the predicted labels,
        and whether the prediction flipped to correct or to wrong.
    """
    if samples not in ("misclassified", "correct", "all"):
        raise ValueError("samples must be 'misclassified', 'correct' or 'all'.")

    feature_sets = _as_feature_sets(features)

    sae_model = sae_model.to(device).eval()
    if hasattr(classifier, "eval"):
        classifier.eval()

    x_all = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    targets = np.asarray(targets).astype(int).reshape(-1)
    maxima = torch.as_tensor(feature_maxima).float().reshape(-1).to(device)
    n = len(x_all)

    if sample_ids is None:
        sample_ids = np.arange(n)
    sample_ids = np.asarray(sample_ids)

    # Which split each row belongs to.
    split_of = np.full(n, "all", dtype=object)
    if splits is not None:
        rows = []
        for name in use_splits:
            if name not in splits:
                raise ValueError(f"Split '{name}' not found. Available: {list(splits)}")
            idx = np.asarray(splits[name]).astype(int)
            split_of[idx] = name
            rows.append(idx)
        rows = np.unique(np.concatenate(rows))
    else:
        rows = np.arange(n)

    with torch.no_grad():
        x = x_all[rows].to(device)
        prob = _positive_probability(classifier, x, positive_label).cpu().numpy()

    # The head must reproduce the stored predictions, otherwise the
    # embeddings and classifier do not belong together.
    if predictions is not None:
        stored = np.asarray(predictions, dtype=float).reshape(-1)[rows]
        max_diff = float(np.max(np.abs(stored - prob))) if len(rows) else 0.0
        if max_diff > prediction_atol:
            raise ValueError(
                "The classifier does not reproduce the stored predictions "
                f"(max difference {max_diff:.2e} > {prediction_atol:.0e}). Check that "
                "the embeddings are the classifier's exact input and that it is in eval mode."
            )

    negative_label = 1 - positive_label
    pred = np.where(prob >= probability_threshold, positive_label, negative_label)
    correct = pred == targets[rows]

    keep = np.ones(len(rows), dtype=bool)
    if samples == "misclassified":
        keep &= ~correct
    elif samples == "correct":
        keep &= correct
    if target_class is not None:
        keep &= targets[rows] == int(target_class)

    rows, x, prob, pred, correct = rows[keep], x[keep], prob[keep], pred[keep], correct[keep]

    if len(rows) == 0:
        print("No samples match the selection; returning an empty table.")
        return pd.DataFrame()

    with torch.no_grad():
        z, _ = sae_model(x)                      # current feature activations (n, n_latent)
        W_dec = sae_model.W_dec.to(device)       # (n_latent, D): one direction per feature

    records = []

    for set_name, feature_ids in feature_sets.items():
        feature_ids = torch.as_tensor(feature_ids, dtype=torch.long, device=device)

        for strength in strengths:
            with torch.no_grad():
                z_f = z[:, feature_ids]
                z_new_f = torch.maximum(z_f, strength * maxima[feature_ids])

                x_new = x + (z_new_f - z_f) @ W_dec[feature_ids]
                prob_new = _positive_probability(classifier, x_new, positive_label).cpu().numpy()

            pred_new = np.where(prob_new >= probability_threshold, positive_label, negative_label)
            correct_new = pred_new == targets[rows]

            for i, row in enumerate(rows):
                records.append({
                    "sample": sample_ids[row],
                    "row": int(row),
                    "split": split_of[row],
                    "feature_set": set_name,
                    "features": feature_ids.tolist(),
                    "strength": float(strength),
                    "target": int(targets[row]),
                    "prob_original": float(prob[i]),
                    "prob_new": float(prob_new[i]),
                    "delta_prob": float(prob_new[i] - prob[i]),
                    "pred_original": int(pred[i]),
                    "pred_new": int(pred_new[i]),
                    "flipped_to_correct": bool(not correct[i] and correct_new[i]),
                    "flipped_to_wrong": bool(correct[i] and not correct_new[i]),
                })

    return pd.DataFrame.from_records(records)
