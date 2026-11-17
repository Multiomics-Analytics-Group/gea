import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.colors import to_rgb
import pandas as pd
import torch
from scipy.stats import fisher_exact
import re
from upsetplot import UpSet, from_contents
from statsmodels.stats.multitest import multipletests
import textwrap


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

def get_annotation_depth(annotation, hierarchy):
    """Return the depth of an annotation in the hierarchy."""

    def search(tree, depth=1):
        for name, children in tree.items():
            if name == annotation:
                return depth

            result = search(children, depth + 1)

            if result is not None:
                return result

        return None

    return search(hierarchy)

def is_descendant(annotation, potential_parent, separator="-->"):
    """Check whether annotation is a descendant of potential_parent."""

    annotation_parts = [
        part.strip() for part in annotation.split(separator)
    ]
    parent_parts = [
        part.strip() for part in potential_parent.split(separator)
    ]

    return (
        len(annotation_parts) > len(parent_parts)
        and annotation_parts[:len(parent_parts)] == parent_parts
    )

def build_feature_annotations(
    concept_feature_pairs,
    numeric_feature_pairs,
    min_categorical_mean=None,
    max_categorical_gap=None,
    min_categorical_test=None,
    min_numeric_mean=None,
    max_numeric_gap=None,
    min_numeric_test=None,
    hierarchy=None,
    score_decimals=4,
):
    """
    Reverse annotation-to-feature dictionaries into feature-to-annotation
    dictionaries, rank candidates, and flag better-performing features.

    Parameters
    ----------
    concept_feature_pairs : dict
        Annotation -> list of feature information dictionaries.
        Each feature dictionary contains:
            - 'feature': SAE feature index.
            - 'validation': dictionary containing validation metrics,
            including 'f1' and optionally 'threshold' and
            'threshold_proportion'.
            - 'test': dictionary containing test metrics, including 'f1'.

    numeric_feature_pairs : dict
        Numeric variable -> list of feature information dictionaries.
        Each feature dictionary contains:
            - 'feature': SAE feature index.
            - 'validation': dictionary containing the validation
            'correlation'.
            - 'test': dictionary containing the test 'correlation'.

    min_categorical_mean, max_categorical_gap, min_categorical_test :
        float or None
        Optional selection criteria for categorical annotations.
        Mean F1 is calculated across validation and test sets, while
        generalisation gap is the absolute difference between them.

    min_numeric_mean, max_numeric_gap, min_numeric_test : float or None
        Optional selection criteria for numeric annotations.
        Mean absolute correlation and generalisation gap are calculated
        using the absolute validation and test correlations.

    hierarchy : optional
        Annotation hierarchy used to prioritise more specific annotations
        when their scores are tied.

    score_decimals : int
        Number of decimal places used to identify tied scores when
        applying the annotation hierarchy.

    Returns
    -------
    dict
        SAE feature -> dictionary containing ranked categorical and
        numeric annotations. Each annotation entry includes its
        validation and test scores, mean score, generalisation gap,
        acceptance status, rank and additional notes.
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

            val = info.get("validation", {}).get("f1", np.nan)
            test = info.get("test", {}).get("f1", np.nan)

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
                "threshold": info.get("validation", {}).get("threshold"),
                "threshold_proportion": info.get(
                    "validation", {}
                ).get("threshold_proportion"),
                "accepted": accepted,
                "notes": [],
            })

    
    # Numeric annotations
    for annotation, feature_list in numeric_feature_pairs.items():
        for info in feature_list:
            feature = info["feature"]

            val = float(
                info.get("validation", {}).get("correlation", np.nan)
            )
            test = float(
                info.get("test", {}).get("correlation", np.nan)
            )

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
            
            if hierarchy is not None:
                i = 0
                while i < len(selected):
                    if not np.isfinite(selected[i]["mean_score"]):
                        i += 1
                        continue

                    score = round(
                        selected[i]["mean_score"],
                        score_decimals,
                    )

                    # Find annotations with the same rounded score
                    tied = []
                    j = i

                    while (
                        j < len(selected)
                        and np.isfinite(selected[j]["mean_score"])
                        and round(
                            selected[j]["mean_score"],
                            score_decimals,
                        ) == score
                    ):
                        tied.append(selected[j])
                        j += 1

                    # Prioritise descendants over their ancestors
                    for entry in tied:
                        annotation = entry["annotation"]

                        entry["_specificity_priority"] = sum(
                            is_descendant(
                                other["annotation"],
                                annotation,
                            )
                            for other in tied
                            if other is not entry
                        )


                    tied.sort(
                        key=lambda entry: entry["_specificity_priority"],
                        reverse=True,
                    )

                    selected[i:j] = tied

                    i = j

            for rank, entry in enumerate(selected, start=1):
                entry["rank"] = rank
                entry.pop("_specificity_priority", None)

            feature_annotations[feature][annotation_type] = selected

    return feature_annotations

def plot_top_feature_annotations(
    feature_annotations,
    annotation_type,
    title,
    top_k=20,
    path=None,
):

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

    ax.set_xlabel(
        "F1 score"
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


def calculate_feature_metrics(
    class_activations,
    feature_annotations,
    class_names=None,
    positive_label=1,
    probability_threshold=0.5,
    accepted_only=True,
    threshold_key="threshold",
    activation_thresholds=(0, 0.15, 0.5, 0.6, 0.8),
):
    """
    Calculate activation, class discrimination and misclassification
    metrics for every eligible annotated SAE feature.

    No features are removed based on activation prevalence.

    Returns
    -------
    dict
        {
            "metadata": dict,
            "features": pd.DataFrame,
            "activation_profiles": dict,
        }
    """

    if class_names is None:
        class_names = {0: "class_0", 1: "class_1"}

    if len(class_names) != 2 or positive_label not in class_names:
        raise ValueError("Exactly two classes must be provided.")

    negative_label = next(
        label for label in class_names if label != positive_label
    )

    positive_name = class_names[positive_label]
    negative_name = class_names[negative_label]

    group_names = [
        f"correct_{positive_name}",
        f"misclassified_{positive_name}",
        f"correct_{negative_name}",
        f"misclassified_{negative_name}",
    ]

    # ---------------------------------------------------------
    # 1. Select the best eligible annotation for each feature.
    # ---------------------------------------------------------

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
                not accepted_only or ann.get("accepted", False)
            )
            and ann.get(threshold_key) is not None
        ]

        if not candidates:
            continue

        best = candidates[0]

        best_annotations[int(feature_id)] = {
            "annotation": best["annotation"],
            "threshold": float(best[threshold_key]),
            "validation_score": best.get("validation_score"),
            "test_score": best.get("test_score"),
            "mean_score": best.get("mean_score"),
            "generalisation_gap": best.get("generalisation_gap"),
            "best_other_feature": best.get("best_other_feature"),
        }

    if not best_annotations:
        raise ValueError("No eligible categorical annotations found.")

    # ---------------------------------------------------------
    # 2. Load activations and normalise using training maxima.
    # ---------------------------------------------------------

    train_maxima = torch.as_tensor(
        class_activations["train_feature_maxima"]
    ).float().cpu().numpy().reshape(-1)

    activation_rows = []
    targets = []
    predictions = []

    for record in class_activations["results"].values():

        activation = torch.as_tensor(
            record["activations"]
        ).float().cpu().numpy().reshape(-1)

        if len(activation) != len(train_maxima):
            raise ValueError(
                "Activation vector and training maxima have "
                "different lengths."
            )

        activation_rows.append(activation)
        targets.append(int(record["target"]))
        predictions.append(float(record["prediction"]))

    if not activation_rows:
        raise ValueError("No sample activations found.")

    activations = np.asarray(activation_rows)
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values.")

    if not np.isin(targets, list(class_names)).all():
        raise ValueError("Unknown target labels found.")

    valid_maxima = (
        np.isfinite(train_maxima) & (train_maxima > 0)
    )

    normalized = np.full_like(activations, np.nan, dtype=float)

    normalized[:, valid_maxima] = (
        activations[:, valid_maxima]
        / train_maxima[valid_maxima]
    )

    # ---------------------------------------------------------
    # 3. Identify correctly classified and misclassified samples.
    # ---------------------------------------------------------

    predicted_labels = np.where(
        predictions >= probability_threshold,
        positive_label,
        negative_label,
    )

    correct = targets == predicted_labels
    misclassified = ~correct

    group_masks = {
        f"correct_{positive_name}": (
            correct & (targets == positive_label)
        ),
        f"misclassified_{positive_name}": (
            misclassified & (targets == positive_label)
        ),
        f"correct_{negative_name}": (
            correct & (targets == negative_label)
        ),
        f"misclassified_{negative_name}": (
            misclassified & (targets == negative_label)
        ),
    }

    group_activations = {
        name: normalized[mask]
        for name, mask in group_masks.items()
    }

    group_sizes = {
        name: int(mask.sum())
        for name, mask in group_masks.items()
    }

    if (
        group_sizes[f"correct_{positive_name}"] == 0
        or group_sizes[f"correct_{negative_name}"] == 0
    ):
        raise ValueError(
            "Both classes must contain correctly classified samples."
        )

    # ---------------------------------------------------------
    # 4. Prepare annotation prevalence.
    # ---------------------------------------------------------

    annotation_prevalence = class_activations.get(
        "annotation_prevalence"
    )

    if annotation_prevalence is not None:
        if not isinstance(annotation_prevalence, pd.DataFrame):
            annotation_prevalence = pd.DataFrame(
                annotation_prevalence
            )

    # ---------------------------------------------------------
    # 5. Calculate metrics for every eligible feature.
    # ---------------------------------------------------------

    feature_rows = []
    activation_profiles = {}

    suffixes = [
        f"_{name}" for name in class_names.values()
    ]

    for feature_id, annotation_info in best_annotations.items():

        if (
            feature_id < 0
            or feature_id >= len(train_maxima)
            or not valid_maxima[feature_id]
        ):
            continue

        threshold = annotation_info["threshold"]

        # Extract finite activation values for each group.
        group_values = {
            name: values[:, feature_id][
                np.isfinite(values[:, feature_id])
            ]
            for name, values in group_activations.items()
        }

        sizes = {
            name: len(values)
            for name, values in group_values.items()
        }

        # Count active samples at annotation-specific threshold.
        active_counts = {
            name: int((values > threshold).sum())
            for name, values in group_values.items()
        }

        # Activation frequencies.
        frequencies = {
            name: (
                active_counts[name] / sizes[name]
                if sizes[name] > 0 else np.nan
            )
            for name in group_names
        }

        # Total counts.
        n_active = sum(active_counts.values())

        n_active_correct = (
            active_counts[f"correct_{positive_name}"]
            + active_counts[f"correct_{negative_name}"]
        )

        n_active_misclassified = (
            active_counts[f"misclassified_{positive_name}"]
            + active_counts[f"misclassified_{negative_name}"]
        )

        n_active_positive = (
            active_counts[f"correct_{positive_name}"]
            + active_counts[f"misclassified_{positive_name}"]
        )

        n_active_negative = (
            active_counts[f"correct_{negative_name}"]
            + active_counts[f"misclassified_{negative_name}"]
        )

        # -----------------------------------------------------
        # A. Class discrimination metrics.
        # -----------------------------------------------------

        fp = frequencies[f"correct_{positive_name}"]
        fn = frequencies[f"correct_{negative_name}"]

        frequency_difference = fp - fn
        absolute_frequency_difference = abs(frequency_difference)

        specificity_score = (
            absolute_frequency_difference / (fp + fn)
            if (fp + fn) > 0 else np.nan
        )

        balanced_accuracy = (
            fp + (1 - fn)
        ) / 2

        # Fisher's exact test: correct positive vs correct negative.
        class_table = [
            [
                active_counts[f"correct_{positive_name}"],
                sizes[f"correct_{positive_name}"]
                - active_counts[f"correct_{positive_name}"],
            ],
            [
                active_counts[f"correct_{negative_name}"],
                sizes[f"correct_{negative_name}"]
                - active_counts[f"correct_{negative_name}"],
            ],
        ]

        class_odds_ratio, class_p_value = fisher_exact(
            class_table,
            alternative="two-sided",
        )

        # -----------------------------------------------------
        # B. Misclassification-related metrics.
        # -----------------------------------------------------

        positive_error_difference = (
            frequencies[f"misclassified_{positive_name}"] - fp
            if sizes[f"misclassified_{positive_name}"] > 0
            else np.nan
        )

        negative_error_difference = (
            frequencies[f"misclassified_{negative_name}"] - fn
            if sizes[f"misclassified_{negative_name}"] > 0
            else np.nan
        )

        # Fisher tests within each true class:
        # Misclassified vs correctly classified samples.

        error_results = {}

        for class_name in [positive_name, negative_name]:

            correct_group = f"correct_{class_name}"
            error_group = f"misclassified_{class_name}"

            if (
                sizes[correct_group] == 0
                or sizes[error_group] == 0
            ):
                error_results[class_name] = {
                    "odds_ratio": np.nan,
                    "p_value": np.nan,
                }
                continue

            error_table = [
                [
                    active_counts[error_group],
                    sizes[error_group] - active_counts[error_group],
                ],
                [
                    active_counts[correct_group],
                    sizes[correct_group] - active_counts[correct_group],
                ],
            ]

            odds_ratio, p_value = fisher_exact(
                error_table,
                alternative="two-sided",
            )

            error_results[class_name] = {
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            }

        # -----------------------------------------------------
        # C. Precision.
        # -----------------------------------------------------

        precisions = {
            name: (
                active_counts[name] / n_active
                if n_active > 0 else np.nan
            )
            for name in group_names
        }

        precision_difference = (
            precisions[f"correct_{positive_name}"]
            - precisions[f"correct_{negative_name}"]
        )

        # -----------------------------------------------------
        # D. Annotation prevalence.
        # -----------------------------------------------------

        annotation = str(annotation_info["annotation"])

        annotation_name = annotation

        for suffix in suffixes:
            if annotation_name.endswith(suffix):
                annotation_name = annotation_name[:-len(suffix)]
                break

        prevalence = {}

        if (
            annotation_prevalence is not None
            and annotation_name in annotation_prevalence.index
        ):
            prevalence_row = annotation_prevalence.loc[
                annotation_name
            ]

            if isinstance(prevalence_row, pd.DataFrame):
                prevalence_row = prevalence_row.iloc[0]

            prevalence = {
                name: prevalence_row.get(name, np.nan)
                for name in group_names
            }
        else:
            prevalence = {
                name: np.nan for name in group_names
            }

        # -----------------------------------------------------
        # E. Activation profiles at all plotting thresholds.
        # -----------------------------------------------------

        profiles = {}

        for name in group_names:

            values = group_values[name]

            profiles[name] = [
                float(np.mean(values > t))
                if len(values) > 0 else np.nan
                for t in activation_thresholds
            ]

        activation_profiles[int(feature_id)] = profiles

        # -----------------------------------------------------
        # F. Store all metrics.
        # -----------------------------------------------------

        row = {
            "feature_id": int(feature_id),
            "annotation": annotation,
            "annotation_name": annotation_name,
            "annotation_threshold": threshold,

            "validation_score": annotation_info["validation_score"],
            "test_score": annotation_info["test_score"],
            "mean_score": annotation_info["mean_score"],
            "generalisation_gap": annotation_info["generalisation_gap"],

            "n_active": n_active,
            "n_inactive": sum(sizes.values()) - n_active,
            "n_active_correct": n_active_correct,
            "n_active_misclassified": n_active_misclassified,
            "n_active_positive": n_active_positive,
            "n_active_negative": n_active_negative,

            "frequency_difference": frequency_difference,
            "absolute_frequency_difference": absolute_frequency_difference,
            "specificity_score": specificity_score,
            "balanced_accuracy": balanced_accuracy,

            "odds_ratio": class_odds_ratio,
            "fisher_p_value": class_p_value,

            "precision_difference": precision_difference,

            "positive_error_frequency_difference": positive_error_difference,
            "negative_error_frequency_difference": negative_error_difference,

            "positive_error_odds_ratio": error_results[
                positive_name
            ]["odds_ratio"],
            "positive_error_p_value": error_results[
                positive_name
            ]["p_value"],

            "negative_error_odds_ratio": error_results[
                negative_name
            ]["odds_ratio"],
            "negative_error_p_value": error_results[
                negative_name
            ]["p_value"],
        }

        for name in group_names:

            row[f"{name}_frequency"] = frequencies[name]
            row[f"{name}_precision"] = precisions[name]
            row[f"{name}_n_active"] = active_counts[name]
            row[f"{name}_n"] = sizes[name]
            row[f"{name}_annotation_prevalence"] = prevalence[name]

        feature_rows.append(row)

    if not feature_rows:
        raise ValueError("No valid annotated features were found.")

    features = pd.DataFrame(feature_rows)

    # ---------------------------------------------------------
    # 6. Multiple-testing correction.
    # ---------------------------------------------------------

    def add_fdr(p_column, q_column):

        valid = features[p_column].notna() & np.isfinite(
            features[p_column].astype(float)
        )

        features[q_column] = np.nan

        if valid.any():

            features.loc[valid, q_column] = multipletests(
                features.loc[valid, p_column].astype(float),
                method="fdr_bh",
            )[1]

    # Separate correction families.
    add_fdr("fisher_p_value", "fdr_p_value")
    add_fdr("positive_error_p_value", "positive_error_fdr")
    add_fdr("negative_error_p_value", "negative_error_fdr")

    return {
        "metadata": {
            "class_names": class_names,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "probability_threshold": probability_threshold,
            "threshold_key": threshold_key,
            "activation_thresholds": list(activation_thresholds),
            "group_names": group_names,
        },
        "features": features,
        "activation_profiles": activation_profiles,
    }


def select_features(
    metrics,
    ranking_metric="absolute_frequency_difference",
    top_k=10,
    ascending=False,
    min_active=None,
    min_active_correct=None,
    max_fdr=None,
    class_direction="both",
):
    """
    Select and rank features from a calculated metrics object.

    Parameters
    ----------
    ranking_metric : str
        Column in metrics["features"] used for ranking.

    ascending : bool
        Whether smaller values should rank first.

    min_active : int or None
        Minimum total active samples across all four groups.

    min_active_correct : int or None
        Minimum active correctly classified samples.

    max_fdr : float or None
        Maximum allowed class-association FDR.

    class_direction : {"both", "positive", "negative"}
        Restrict frequency-difference rankings to features
        favouring one class.
    """

    features = metrics["features"].copy()

    if ranking_metric not in features.columns:
        raise ValueError(
            f"Unknown ranking metric: {ranking_metric}"
        )

    if min_active is not None:
        features = features[
            features["n_active"] >= min_active
        ]

    if min_active_correct is not None:
        features = features[
            features["n_active_correct"] >= min_active_correct
        ]

    if max_fdr is not None:
        features = features[
            features["fdr_p_value"].notna()
            & (features["fdr_p_value"] <= max_fdr)
        ]

    if class_direction == "positive":
        features = features[
            features["frequency_difference"] > 0
        ]

    elif class_direction == "negative":
        features = features[
            features["frequency_difference"] < 0
        ]

    elif class_direction != "both":
        raise ValueError(
            "class_direction must be 'both', 'positive' or 'negative'."
        )

    features = features.dropna(subset=[ranking_metric])

    # Use FDR as a secondary sorting criterion when available.
    sort_columns = [ranking_metric]
    sort_order = [ascending]

    if ranking_metric != "fdr_p_value":
        sort_columns.append("fdr_p_value")
        sort_order.append(True)

    features = features.sort_values(
        sort_columns,
        ascending=sort_order,
        na_position="last",
    )

    return features.head(top_k).reset_index(drop=True)


def plot_selected_features(
    metrics,
    selected_features,
    figsize_per_feature=(5, 3.5),
    ylim=(0, 1),
    output_path=None,
):
    """
    Plot activation profiles and annotation prevalence
    for selected SAE features.

    Parameters
    ----------
    metrics : dict
        Output from calculate_feature_metrics().

    selected_features : pd.DataFrame
        Output from select_features().

    Returns
    -------
    None
    """

    if selected_features.empty:
        raise ValueError("No features selected for plotting.")

    metadata = metrics["metadata"]

    positive_name = metadata["class_names"][
        metadata["positive_label"]
    ]

    negative_name = metadata["class_names"][
        metadata["negative_label"]
    ]

    thresholds = metadata["activation_thresholds"]

    profiles = metrics["activation_profiles"]

    n_features = len(selected_features)
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

    x = np.arange(len(thresholds))
    width = 0.2

    positive_colour = "#CEAFE1"
    negative_colour = "#BADBC6"

    group_plot_info = [
        (f"correct_{positive_name}", -1.5 * width,
         positive_colour, None),
        (f"misclassified_{positive_name}", -0.5 * width,
         positive_colour, "//"),
        (f"correct_{negative_name}", 0.5 * width,
         negative_colour, None),
        (f"misclassified_{negative_name}", 1.5 * width,
         negative_colour, "//"),
    ]

    for plot_idx, (_, row) in enumerate(
        selected_features.iterrows()
    ):

        ax = axes[plot_idx]

        feature_id = int(row["feature_id"])

        feature_profiles = profiles[feature_id]

        # Plot activation frequencies.
        for group_name, offset, colour, hatch in group_plot_info:

            ax.bar(
                x + offset,
                feature_profiles[group_name],
                width,
                label=group_name.replace("_", " ").capitalize(),
                color=colour,
                hatch=hatch,
                edgecolor="grey",
                linewidth=0.8,
            )

        # Plot annotation prevalence points.
        threshold = row["annotation_threshold"]

        threshold_idx = int(
            np.argmin(
                np.abs(np.asarray(thresholds, dtype=float) - threshold)
            )
        )

        for group_name, offset, colour, _ in group_plot_info:

            prevalence = row[
                f"{group_name}_annotation_prevalence"
            ]

            if pd.notna(prevalence):

                ax.scatter(
                    threshold_idx + offset,
                    prevalence,
                    color=colour,
                    marker=".",
                    edgecolor="grey",
                    linewidth=0.7,
                    s=75,
                    zorder=10,
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

        # Plot title and metrics.
        annotation = textwrap.fill(
            str(row["annotation"]),
            width=40,
        )

        ax.set_title(
            f"Feature {feature_id}: {annotation}",
            #fontsize=10,
            pad=8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in thresholds])

        ax.set_xlabel("Normalised activation threshold")
        ax.set_ylabel("Fraction of samples activated")
        ax.set_ylim(ylim)

        ax.legend()

    for ax in axes[n_features:]:
        ax.set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()
    plt.close(fig)
