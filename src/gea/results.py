import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.colors import to_rgb
import pandas as pd
import torch
from scipy.stats import fisher_exact
import re
from upsetplot import UpSet, from_contents

def summarize_feature_activity(frequency, threshold=0.0001):
    frequency = frequency.detach().cpu().numpy()

    dead = frequency == 0
    rare = (frequency > 0) & (frequency < threshold)
    active = frequency > 0

    return {
        "n_features": len(frequency),
        "n_dead": int(dead.sum()),
        "pct_dead": float(dead.mean() * 100),
        "dead_features": np.flatnonzero(dead).tolist(),
        "n_rare": int(rare.sum()),
        "pct_rare": float(rare.mean() * 100),
        "rare_features": np.flatnonzero(rare).tolist(),
        "n_active": int(active.sum()),
        "pct_active": float(active.mean() * 100),
        "active_features": np.flatnonzero(active).tolist(),
    }

def plot_feature_activity_upset(
    annotation_nodes,
    feature_key="dead_features",
    split_names=("Train", "Validation", "Test"),
    title=None,
    output_path=None,
    figsize=(9, 5),
):
    """Plot an UpSet plot comparing feature activity across data splits."""

    palette = {
        "train": "#BADBC6",
        "val": "#CEAFE1",
        "test": "#F2C284"
    }
    valid_keys = {"dead_features", "rare_features", "active_features"}
    if feature_key not in valid_keys:
        raise ValueError(f"feature_key must be one of {valid_keys}")

    frequency_keys = (
        "frequency_stats_train",
        "frequency_stats_val",
        "frequency_stats_test",
    )

    summaries = {
        split: summarize_feature_activity(annotation_nodes[key])
        for split, key in zip(split_names, frequency_keys)
    }

    feature_sets = {
        split: set(summary[feature_key])
        for split, summary in summaries.items()
    }

    upset_data = from_contents(feature_sets)

    fig = plt.figure(figsize=figsize)
    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
    )
    
    axes = upset.plot(fig=fig)

    # Colour the left set-size bars
    split_palette = {
        "Train": palette["train"],
        "Validation": palette["val"],
        "Test": palette["test"],
    }

    totals_ax = axes["totals"]
    totals_ax.grid(False, axis="x")
    axes["intersections"].grid(False)

    for bar, label in zip(totals_ax.patches, totals_ax.get_yticklabels()):
        split_name = label.get_text()
        bar.set_facecolor(split_palette[split_name])
        bar.set_edgecolor("none")
        bar.set_linewidth(0)

    # Fix one-element array positions for Matplotlib compatibility
    for ax in fig.axes:
        for text in ax.findobj(match=plt.Text):
            x, y = text.get_position()

            if np.ndim(x) == 1 and len(x) == 1:
                x = x.item()

            if np.ndim(y) == 1 and len(y) == 1:
                y = y.item()

            text.set_position((x, y))

    if title is not None:
        fig.suptitle(title)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)

    plt.show()

    return fig, summaries

def build_feature_annotations(
    concept_feature_pairs,
    numeric_feature_pairs,
    min_categorical_mean=None,
    max_categorical_gap=None,
    min_categorical_test=None,
    min_numeric_mean=None,
    max_numeric_gap=None,
    min_numeric_test=None,
):
    """
    Reverse annotation-to-feature dictionaries into feature-to-annotation
    dictionaries, rank candidates, and flag better-performing features.

    Parameters
    ----------
    concept_feature_pairs : dict
        Annotation -> list of feature information dictionaries.
        Requires 'feature', 'score', and 'test'['score'] (F1 or MCC).
        Results saved before the score option was added use 'f1'
        instead of 'score'; both are accepted.

    numeric_feature_pairs : dict
        Numeric variable -> list of feature information dictionaries.
        Requires 'feature', 'correlation', and 'test_correlation'.

    Threshold arguments : float or None
        Optional selection criteria. None disables that criterion.

    Returns
    -------
    dict
        SAE feature -> ranked categorical and numeric annotations.
    """

    candidates = {}

    def add_candidate(feature, entry):
        candidates.setdefault(feature, []).append(entry)

    def passes_filters(mean, gap, test_score, min_mean, max_gap, min_test):
        if not np.isfinite(mean) or not np.isfinite(gap):
            return False
        if min_mean is not None and mean < min_mean:
            return False
        if max_gap is not None and gap > max_gap:
            return False
        if min_test is not None:
            if not np.isfinite(test_score) or test_score < min_test:
                return False
        return True

    # Categorical annotations    
    for annotation, feature_list in concept_feature_pairs.items():
        for info in feature_list:
            feature = info["feature"]

            test_info = info.get("test", {})
            val = info.get("score", info.get("f1", np.nan))
            test = test_info.get("score", test_info.get("f1", np.nan))

            val = float(val)
            test = float(test)

            if np.isfinite(val) and np.isfinite(test):
                mean = (val + test) / 2
                gap = abs(val - test)
            else:
                mean = np.nan
                gap = np.nan

            accepted = passes_filters(
                mean,
                gap,
                test,
                min_categorical_mean,
                max_categorical_gap,
                min_categorical_test,
            )

            add_candidate(feature, {
                "annotation": annotation,
                "type": "categorical",
                "validation_score": val,
                "test_score": test,
                "mean_score": mean,
                "generalisation_gap": gap,
                "threshold": info.get("threshold"),
                "threshold_proportion": info.get(
                    "threshold_proportion"
                ),
                "accepted": accepted,
                "notes": [],
            })

    
    # Numeric annotations
    for annotation, feature_list in numeric_feature_pairs.items():
        for info in feature_list:
            feature = info["feature"]

            val = float(info.get("correlation", np.nan))
            test = float(info.get("test_correlation", np.nan))

            # Rank correlations by magnitude, retaining their signs.
            if np.isfinite(val) and np.isfinite(test):
                mean = (abs(val) + abs(test)) / 2
                gap = abs(abs(val) - abs(test))
            else:
                mean = np.nan
                gap = np.nan

            accepted = passes_filters(
                mean,
                gap,
                abs(test) if np.isfinite(test) else np.nan,
                min_numeric_mean,
                max_numeric_gap,
                min_numeric_test,
            )

            notes = []
            if (
                np.isfinite(val)
                and np.isfinite(test)
                and val * test < 0
            ):
                notes.append(
                    "Validation and test correlations have "
                    "opposite directions."
                )

            add_candidate(feature, {
                "annotation": annotation,
                "type": "numeric",
                "validation_score": val,
                "test_score": test,
                "mean_score": mean,
                "generalisation_gap": gap,
                "accepted": accepted,
                "notes": notes,
            })

    # Compare each annotation across all SAE features
    best_feature_per_annotation = {}

    for feature, entries in candidates.items():
        for entry in entries:
            if not entry["accepted"]:
                continue
            if not np.isfinite(entry["mean_score"]):
                continue

            key = (entry["type"], entry["annotation"])
            current_best = best_feature_per_annotation.get(key)

            if (
                current_best is None
                or entry["mean_score"] > current_best["mean_score"]
            ):
                best_feature_per_annotation[key] = {
                    "feature": feature,
                    "mean_score": entry["mean_score"],
                }

    for feature, entries in candidates.items():
        for entry in entries:
            key = (entry["type"], entry["annotation"])
            best = best_feature_per_annotation.get(key)

            if best is not None and best["feature"] != feature:
                if best["mean_score"] > entry["mean_score"]:
                    entry["notes"].append(
                        f"Feature {best['feature']} represents this "
                        f"annotation better (mean score "
                        f"{best['mean_score']:.3f} vs "
                        f"{entry['mean_score']:.3f})."
                    )
                    entry["best_other_feature"] = best["feature"]
                else:
                    entry["best_other_feature"] = None
            else:
                entry["best_other_feature"] = None

    # Rank separately by annotation type within each feature
    feature_annotations = {}

    for feature, entries in candidates.items():
        feature_annotations[feature] = {}

        for annotation_type in ("categorical", "numeric"):
            selected = [
                entry for entry in entries
                if entry["type"] == annotation_type
            ]

            selected.sort(
                key=lambda entry: (
                    entry["accepted"],
                    entry["mean_score"]
                    if np.isfinite(entry["mean_score"])
                    else -np.inf,
                ),
                reverse=True,
            )

            for rank, entry in enumerate(selected, start=1):
                entry["rank"] = rank

            feature_annotations[feature][annotation_type] = selected

    return feature_annotations

def plot_top_feature_annotations(
    feature_annotations,
    annotation_type,
    title,
    top_k=20,
    path=None,
    score="f1",
):
    """score : "f1" or "mcc", the score used for categorical annotations
    (only changes the x-axis label)."""

    candidates = []

    for feature, annotations in feature_annotations.items():
        entries = [
            item for item in annotations[annotation_type]
            if item["accepted"]
            and np.isfinite(item["validation_score"])
            and np.isfinite(item["test_score"])
        ]

        entries.sort(
            key=lambda item: item["mean_score"],
            reverse=True,
        )

        if entries:
            candidates.append({
                "feature": feature,
                "entries": entries,
            })

    # Process features from highest to lowest best-candidate mean score.
    candidates.sort(
        key=lambda item: item["entries"][0]["mean_score"],
        reverse=True,
    )

    # Assign each annotation to only one feature.
    used_annotations = set()
    results = []

    for item in candidates:
        best = next(
            (
                entry for entry in item["entries"]
                if entry["annotation"] not in used_annotations
            ),
            None,
        )

        if best is None:
            continue

        used_annotations.add(best["annotation"])

        results.append({
            "feature": item["feature"],
            "annotation": best["annotation"],
            "validation": best["validation_score"],
            "test": best["test_score"],
            "mean": best["mean_score"],
        })

        if len(results) >= top_k:
            break

    if not results:
        print(f"No accepted {annotation_type} annotations found.")
        return

    # Highest-ranked feature at the top.
    results.sort(
        key=lambda item: item["validation"],
        reverse=True,
    )
    results.reverse()

    labels = [
        f"F{r['feature']}: {r['annotation']}"
        for r in results
    ]

    y = np.arange(len(results))

    validation = [r["validation"] for r in results]
    test = [r["test"] for r in results]

    fig, ax = plt.subplots(
        figsize=(11, max(6, len(results) * 0.45))
    )

    lightorange = "#F2C284"
    purple = "#CEAFE1"
    background = (0.92, 0.92, 0.92)

    bar_height = 0.25
    offset = 0.17
    radius = 0.05

    max_value = max(
        max(
            abs(v) if annotation_type == "numeric" else v
            for v in validation + test
        ),
        1.0,
    )

    def add_rounded_bar(y_pos, value, color):
        # Background track.
        rgb = to_rgb(color)
        ax.add_patch(
            FancyBboxPatch(
                (0, y_pos - bar_height / 2),
                max_value,
                bar_height,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=background,
                edgecolor="none",
                zorder=1,
            )
        )

        negative = annotation_type == "numeric" and value < 0

        # Actual score.
        ax.add_patch(
            FancyBboxPatch(
                (0, y_pos - bar_height / 2),
                abs(value) if annotation_type == "numeric" else value,
                bar_height,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=(*rgb, 1.0),
                #edgecolor=(*rgb, 1.0),
                edgecolor="#817D7D" if negative else color,
                hatch="///" if negative else None,
                zorder=2,
            )
        )
        
        ax.text(
            1.015,
            y_pos,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            zorder=3,
        )

    for i, (val, tst) in enumerate(zip(validation, test)):

        add_rounded_bar(
            i + offset,
            val,
            lightorange,
        )

        add_rounded_bar(
            i - offset,
            tst,
            purple,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    score_labels = {"f1": "F1 score", "mcc": "MCC"}

    ax.set_xlabel(
        score_labels.get(score, score)
        if annotation_type == "categorical"
        else "Absolute Spearman correlation"
    )

    ax.set_title(
        title
    )

    ax.set_xlim(0, max_value * 1.05)
    ax.set_ylim(-0.5, len(results) - 0.5)

    legend_handles=[
        Patch(facecolor=lightorange, alpha=1, label="Validation"),
        Patch(facecolor=purple, alpha=1, label="Test"),
    ]

    has_negative = (
        annotation_type == "numeric"
        and any(v < 0 for v in validation + test)
    )

    if has_negative:
        legend_handles.append(
            Patch(
                facecolor="white",
                edgecolor="#817D7D",
                hatch="///",
                label="Negative correlation",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.01)
    )

    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Remove unnecessary borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, bbox_inches="tight", dpi=300)

    plt.show()
    plt.close(fig)

def plot_discriminative_feature_proportions(
    class_activations,
    feature_annotations,
    class_names=None,
    positive_label=1,
    top_k=10,
    probability_threshold=0.5,
    accepted_only=True,
    figsize_per_feature=(5, 3.5),
    output_path=None,
    threshold_key="threshold",
    activation_thresholds=(0, 0.15, 0.5, 0.6, 0.8),
    min_active=10,
    ylim=(0, 1),
):
    """Plot activation-frequency histograms for discriminative SAE features.

    Features are ranked by Fisher's exact test p-value and the absolute
    difference in activation frequency between correctly classified
    samples from the two classes.

    Parameters
    ----------
    class_activations : dict
        Results containing training feature maxima and sample-level
        activations, targets and predictions.

    feature_annotations : dict
        SAE feature -> ranked categorical and numeric annotations.

    class_names : dict or None
        Mapping between class labels and class names.
        Defaults to {0: "class_0", 1: "class_1"}.

    positive_label : int
        Class represented by probabilities above the classification
        threshold.

    probability_threshold : float
        Threshold used to convert probabilities into predicted labels.

    Other parameters
    ----------------
    See function signature.

    Returns
    -------
    pd.DataFrame
        Summary statistics for the selected features.
    """

    if class_names is None:
        class_names = {
            0: "class_0",
            1: "class_1",
        }

    if len(class_names) != 2 or positive_label not in class_names:
        raise ValueError(
            "class_names must contain exactly two classes, "
            "including positive_label."
        )

    negative_label = next(
        label for label in class_names if label != positive_label
    )

    positive_name = class_names[positive_label]
    negative_name = class_names[negative_label]

    # 1. Select the best eligible annotation for each feature.
    best_annotations = {}

    for feature_id, annotation_groups in feature_annotations.items():

        candidates = annotation_groups.get("categorical", [])

        candidates = sorted(
            candidates,
            key=lambda ann: ann.get("rank", float("inf")),
        )

        candidates = [
            ann for ann in candidates
            if (
                not accepted_only
                or ann.get("accepted", False)
            )
            and ann.get(threshold_key) is not None
        ]

        if not candidates:
            continue

        best = candidates[0]

        best_annotations[int(feature_id)] = {
            "annotation": best["annotation"],
            "threshold": float(best[threshold_key]),
            "validation_score": best["validation_score"],
            "test_score": best["test_score"],
            "mean_score": best["mean_score"],
            "generalisation_gap": best["generalisation_gap"],
            "best_other_feature": best.get("best_other_feature"),
        }

    if not best_annotations:
        raise ValueError("No eligible categorical annotations were found.")

    # 2. Load and normalise activations using training maxima.
    train_feature_maxima = torch.as_tensor(
        class_activations["train_feature_maxima"]
    ).float().cpu().numpy().reshape(-1)

    activation_rows = []
    targets = []
    predictions = []

    for record in class_activations["results"].values():

        activation = torch.as_tensor(
            record["activations"]
        ).float().cpu().numpy().reshape(-1)

        if len(activation) != len(train_feature_maxima):
            raise ValueError(
                "Activation vector and training maxima have "
                "different lengths."
            )

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

    valid_maxima = (
        np.isfinite(train_feature_maxima)
        & (train_feature_maxima > 0)
    )

    normalized_activations = np.full_like(
        activations,
        np.nan,
        dtype=float,
    )

    normalized_activations[:, valid_maxima] = (
        activations[:, valid_maxima]
        / train_feature_maxima[valid_maxima]
    )

    # 3. Separate correctly and incorrectly classified samples.
    predicted_labels = np.where(
        predictions >= probability_threshold,
        positive_label,
        negative_label,
    )

    correctly_classified = targets == predicted_labels
    incorrectly_classified = targets != predicted_labels

    group_masks = {
        f"correct_{positive_name}": (
            correctly_classified & (targets == positive_label)
        ),
        f"misclassified_{positive_name}": (
            incorrectly_classified & (targets == positive_label)
        ),
        f"correct_{negative_name}": (
            correctly_classified & (targets == negative_label)
        ),
        f"misclassified_{negative_name}": (
            incorrectly_classified & (targets == negative_label)
        ),
    }

    if (
        not group_masks[f"correct_{positive_name}"].any()
        or not group_masks[f"correct_{negative_name}"].any()
    ):
        raise ValueError(
            "Both classes must contain correctly classified samples."
        )

    group_activations = {
        name: normalized_activations[mask]
        for name, mask in group_masks.items()
    }

    # 4. Rank annotated features.
    annotation_prevalence = class_activations["annotation_prevalence"]

    if not isinstance(annotation_prevalence, pd.DataFrame):
        annotation_prevalence = pd.DataFrame(annotation_prevalence)

    annotated_features = np.array(
        sorted(best_annotations), dtype=int
    )

    valid_features = annotated_features[
        (annotated_features >= 0)
        & (annotated_features < len(train_feature_maxima))
    ]

    valid_features = valid_features[
        valid_maxima[valid_features]
    ]

    feature_scores = {}

    for feature_id in valid_features:

        annotation_info = best_annotations[feature_id]
        threshold = annotation_info["threshold"]

        # Extract finite activations for each group.
        group_values = {
            name: values[:, feature_id][
                np.isfinite(values[:, feature_id])
            ]
            for name, values in group_activations.items()
        }

        correct_positive = group_values[
            f"correct_{positive_name}"
        ]

        correct_negative = group_values[
            f"correct_{negative_name}"
        ]

        if len(correct_positive) == 0 or len(correct_negative) == 0:
            continue

        active_counts = {
            name: int((values > threshold).sum())
            for name, values in group_values.items()
        }

        group_sizes = {
            name: len(values)
            for name, values in group_values.items()
        }

        n_active = (
            active_counts[f"correct_{positive_name}"]
            + active_counts[f"correct_{negative_name}"]
        )

        n_active_total = sum(active_counts.values())

        if n_active < min_active:
            continue

        # Activation frequencies.
        frequencies = {
            name: (
                active_counts[name] / group_sizes[name]
                if group_sizes[name] > 0
                else np.nan
            )
            for name in group_values
        }

        positive_frequency = frequencies[
            f"correct_{positive_name}"
        ]

        negative_frequency = frequencies[
            f"correct_{negative_name}"
        ]

        frequency_difference = (
            positive_frequency - negative_frequency
        )

        # Precision among all active samples.
        precisions = {
            name: (
                active_counts[name] / n_active_total
                if n_active_total > 0
                else np.nan
            )
            for name in group_values
        }

        # Fisher's exact test using correctly classified samples.
        contingency_table = [
            [
                active_counts[f"correct_{positive_name}"],
                group_sizes[f"correct_{positive_name}"]
                - active_counts[f"correct_{positive_name}"],
            ],
            [
                active_counts[f"correct_{negative_name}"],
                group_sizes[f"correct_{negative_name}"]
                - active_counts[f"correct_{negative_name}"],
            ],
        ]

        odds_ratio, p_value = fisher_exact(
            contingency_table,
            alternative="two-sided",
        )

        feature_scores[int(feature_id)] = {
            "annotation": annotation_info["annotation"],
            "threshold": threshold,
            "frequencies": frequencies,
            "precisions": precisions,
            "frequency_difference": frequency_difference,
            "absolute_frequency_difference": abs(
                frequency_difference
            ),
            "active_counts": active_counts,
            "group_sizes": group_sizes,
            "n_active": n_active,
            "odds_ratio": odds_ratio,
            "fisher_p_value": p_value,
        }

    if not feature_scores:
        raise ValueError(
            "No valid categorical annotations met the selection criteria."
        )

    # Select top features.
    top_features = sorted(
        feature_scores,
        key=lambda feature_id: (
            feature_scores[feature_id]["fisher_p_value"],
            -feature_scores[feature_id]["absolute_frequency_difference"],
        ),
    )[:top_k]

    # 5. Plot one grouped bar chart per feature.
    n_features = len(top_features)
    n_cols = min(2, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            figsize_per_feature[0] * n_cols,
            figsize_per_feature[1] * n_rows,
        ),
        squeeze=False,
    )

    axes = axes.flatten()

    thresholds = list(activation_thresholds)
    x = np.arange(len(thresholds))
    width = 0.2

    plt.rcParams["hatch.color"] = "grey"
    plt.rcParams["hatch.linewidth"] = 0.8

    positive_colour = "#CEAFE1"
    negative_colour = "#BADBC6"

    group_plot_info = [
        (f"correct_{positive_name}", -1.5 * width, positive_colour, None),
        (f"misclassified_{positive_name}", -0.5 * width, positive_colour, "//"),
        (f"correct_{negative_name}", 0.5 * width, negative_colour, None),
        (f"misclassified_{negative_name}", 1.5 * width, negative_colour, "//"),
    ]

    summary = []

    for plot_idx, feature_id in enumerate(top_features):

        ax = axes[plot_idx]
        stats = feature_scores[feature_id]

        group_values = {
            name: values[:, feature_id][
                np.isfinite(values[:, feature_id])
            ]
            for name, values in group_activations.items()
        }

        group_frequencies = {
            name: [
                np.mean(values > threshold)
                if len(values) > 0 else 0
                for threshold in thresholds
            ]
            for name, values in group_values.items()
        }

        for group_name, offset, colour, hatch in group_plot_info:

            ax.bar(
                x + offset,
                group_frequencies[group_name],
                width,
                label=group_name.replace("_", " ").capitalize(),
                color=colour,
                hatch=hatch,
                edgecolor="grey",
                linewidth=0.8,
            )

        annotation = stats["annotation"]
        annotation_threshold = stats["threshold"]

        # Remove class-specific suffixes from annotation names.
        suffixes = [
            re.escape(f"_{name}")
            for name in class_names.values()
        ]

        suffix_pattern = (
            r"(?:"
            + "|".join(suffixes)
            + r"|_presence)$"
        )

        annotation_name = re.sub(
            suffix_pattern,
            "",
            str(annotation),
        )


        threshold_idx = int(
            np.argmin(
                np.abs(
                    np.asarray(thresholds, dtype=float)
                    - annotation_threshold
                )
            )
        )

        ax.axvspan(
            threshold_idx - 0.5,
            threshold_idx + 0.5,
            color="lightgrey",
            alpha=0.18,
            zorder=0,
        )

        ax.text(
            threshold_idx,
            0.98,
            "Annotation\nthreshold",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )


        if annotation_name in annotation_prevalence.index:

            prevalence = annotation_prevalence.loc[annotation_name]

            for group_name, offset, colour, _ in group_plot_info:

                point_y = prevalence.get(group_name, np.nan)

                if pd.notna(point_y):
                    ax.scatter(
                        threshold_idx + offset,
                        point_y,
                        color=colour,
                        marker=".",
                        edgecolor="grey",
                        linewidth=0.7,
                        s=75,
                        zorder=10,
                    )

        ax.set_title(
            f"Feature {feature_id}: {annotation}\n"
            f"Fisher p = {stats['fisher_p_value']:.2e}; "
            f"OR = {stats['odds_ratio']:.2g}\n"
            f"{positive_name.capitalize()} precision = "
            f"{stats['precisions'][f'correct_{positive_name}']:.2f}; "
            f"{negative_name.capitalize()} precision = "
            f"{stats['precisions'][f'correct_{negative_name}']:.2f}"
        )

        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in thresholds])
        ax.set_xlabel("Normalised activation threshold")
        ax.set_ylabel("Fraction of samples activated")
        ax.set_ylim(ylim)
        ax.legend()

        result = {
            "feature_id": int(feature_id),
            "annotation": annotation,
            "annotation_threshold": annotation_threshold,
            "frequency_difference": stats["frequency_difference"],
            "absolute_frequency_difference": stats[
                "absolute_frequency_difference"
            ],
            "n_active": stats["n_active"],
            "odds_ratio": stats["odds_ratio"],
            "fisher_p_value": stats["fisher_p_value"],
        }

        for group_name in group_masks:
            result[f"{group_name}_frequency"] = (
                stats["frequencies"][group_name]
            )
            result[f"{group_name}_precision"] = (
                stats["precisions"][group_name]
            )
            result[f"{group_name}_n_active"] = (
                stats["active_counts"][group_name]
            )
            result[f"{group_name}_n"] = (
                stats["group_sizes"][group_name]
            )

        summary.append(result)

    for ax in axes[n_features:]:
        ax.set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)

    plt.show()
    plt.close(fig)

    return pd.DataFrame(summary)