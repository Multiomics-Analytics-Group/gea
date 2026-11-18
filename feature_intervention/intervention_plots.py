"""Feature intervention: plots of the results of feature_intervention.py."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ── Feature intervention plots ─────────────────────────────────────────────────
#
# These read the table produced by feature_intervention.py: the output of
# intervention.feature_intervention plus a "condition" column (see
# INTERVENTION_CONDITIONS) and a "class" column (class name of the samples).

INTERVENTION_CONDITIONS = {
    "single": "true-class feature",
    "single_random": "random feature",
    "set": "true-class set",
    "set_random": "random set",
    "set_wrong": "wrong-class set",
}

_INTERVENTION_COLOURS = {
    "selected": "#2a78d6",   # reference palette slot 1 (blue)
    "wrong": "#eb6834",      # reference palette slot 2 (orange)
    "random": "#a9a8a2",     # neutral grey: context, not a series
    "ink": "#0b0b0b",
    "ink_2": "#52514e",
    "grid": "#e4e3df",
}


def _style_intervention_axes(ax, grid_axis="y"):
    c = _INTERVENTION_COLOURS
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["ink_2"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["ink_2"], labelsize=9)
    ax.grid(axis=grid_axis, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _percent_axis(axis):
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def _random_flip_summary(random_results, strengths):
    """Mean, lowest and highest flip-to-correct rate over random draws, per strength.

    Pass the random rows of ONE class: each draw is then one rate over that
    class's samples, and the range is simply the lowest and highest of those
    draws. Every draw covers the same samples, so the mean over draws equals
    the pooled mean, the same statistic as for the selected features.
    """
    per_draw = (
        random_results.groupby(["strength", "feature_set"])["flipped_to_correct"]
        .mean().unstack("strength").reindex(columns=strengths)
    )
    return per_draw.mean(), per_draw.min(), per_draw.max()


def plot_intervention_dose_response(results, output_path=None, title=None):
    """All selected features of a class added together: fraction of the class's
    misclassified samples flipped to the correct class, against strength. One
    panel per true class.

    Each point is the share of that class's samples flipped by its one set of
    selected features (blue), by the other class's selected features (orange,
    a direction check), or by random sets of the same size (grey: mean over
    draws, band from the lowest to the highest draw). One feature at a time is
    shown in plot_intervention_per_feature.
    """
    c = _INTERVENTION_COLOURS
    cond = INTERVENTION_CONDITIONS
    strengths = sorted(results["strength"].unique())
    x = np.arange(len(strengths))
    classes = sorted(results["class"].unique())

    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 4),
                             sharey=True, squeeze=False)

    for ax, cls in zip(axes[0], classes):
        d = results[results["class"] == cls]
        n = d["sample"].nunique()

        rnd = d[d["condition"] == cond["set_random"]]
        if not rnd.empty:
            mean, low, high = _random_flip_summary(rnd, strengths)
            n_draws = rnd["feature_set"].nunique()
            ax.fill_between(x, low, high, color=c["random"], alpha=0.3, linewidth=0,
                            label=f"Random sets (lowest to highest of {n_draws} draws)")
            ax.plot(x, mean, color=c["random"], linewidth=2, marker="o",
                    markersize=7, label="Random sets (mean)")

        sel = d[d["condition"] == cond["set"]]
        if not sel.empty:
            rate = sel.groupby("strength")["flipped_to_correct"].mean().reindex(strengths)
            ax.plot(x, rate, color=c["selected"], linewidth=2, marker="o", markersize=8,
                    label="Selected true-class features")
            for xi, v in zip(x, rate):
                ax.annotate(f"{v:.0%}", (xi, v), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9, color=c["ink"])

        wr = d[d["condition"] == cond["set_wrong"]]
        if not wr.empty:
            rate = wr.groupby("strength")["flipped_to_correct"].mean().reindex(strengths)
            ax.plot(x, rate, color=c["wrong"], linewidth=2, marker="o", markersize=8,
                    label="Other class's selected features")

        ax.set_title(f"Truly {cls} (n={n})", fontsize=11, color=c["ink"], loc="left")
        ax.set_xticks(x, [f"{s:g}×" for s in strengths])
        ax.set_xlabel("Strength (× feature's training maximum)", fontsize=9, color=c["ink_2"])
        ax.set_ylim(-0.03, 1.1)
        _percent_axis(ax.yaxis)
        _style_intervention_axes(ax)

    axes[0, 0].set_ylabel("Misclassified samples flipped to correct", fontsize=9, color=c["ink_2"])

    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title or "All selected features of a class added together",
                 fontsize=12, color=c["ink"], x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_intervention_per_feature(results, strengths=None, sort_strength=1.0,
                                  feature_labels=None, output_path=None):
    """Each selected feature on its own: one panel per strength, same rows in
    every panel.

    Each blue dot is the share of that feature's class's misclassified samples
    it flips on its own. Each row's grey band runs from the lowest to the
    highest random single feature tested on the same class's samples; the grey
    tick is their mean.

    strengths : list of float or None
        Strengths to show (default: all in results).
    sort_strength : float
        Rows are sorted by the flip rate at this strength (default 1; the
        middle strength if 1 was not run).
    feature_labels : dict or None
        {feature id: label}, e.g. the feature's annotation name.
    """
    c = _INTERVENTION_COLOURS
    cond = INTERVENTION_CONDITIONS
    feature_labels = feature_labels or {}

    sel_all = results[results["condition"] == cond["single"]]
    if sel_all.empty:
        print("No single-feature results to plot.")
        return None

    if strengths is None:
        strengths = sorted(sel_all["strength"].unique())
    if not np.isclose(strengths, sort_strength).any():
        sort_strength = strengths[len(strengths) // 2]

    # Row order: flip rate at sort_strength (lowest at the bottom).
    rows = (
        sel_all[np.isclose(sel_all["strength"], sort_strength)]
        .groupby(["feature_set", "class"])["flipped_to_correct"].mean()
        .sort_values().reset_index()[["feature_set", "class"]]
    )
    y = np.arange(len(rows))

    def short(text, width=36):
        text = str(text)
        return text if len(text) <= width else text[: width - 1] + "…"

    labels = []
    for feature, cls in zip(rows["feature_set"], rows["class"]):
        name = feature_labels.get(int(feature)) if str(feature).isdigit() else None
        labels.append(f"F{feature}" + (f" · {short(name)}" if name else "") + f" ({cls})")

    fig, axes = plt.subplots(1, len(strengths), figsize=(4.0 * len(strengths) + 3.5,
                                                         0.45 * len(rows) + 1.8),
                             sharey=True, squeeze=False)
    n_draws = 0

    for ax, strength in zip(axes[0], strengths):
        d = results[np.isclose(results["strength"], strength)]
        rates = (
            d[d["condition"] == cond["single"]]
            .groupby(["feature_set", "class"])["flipped_to_correct"].agg(["mean", "size"])
        )
        rnd = (
            d[d["condition"] == cond["single_random"]]
            .groupby(["class", "feature_set"])["flipped_to_correct"].mean()
        )

        for yi, (feature, cls) in zip(y, zip(rows["feature_set"], rows["class"])):
            if cls in rnd.index.get_level_values("class"):
                draws = rnd.loc[cls]
                n_draws = len(draws)
                ax.fill_betweenx([yi - 0.32, yi + 0.32], draws.min(), draws.max(),
                                 color=c["random"], alpha=0.3, linewidth=0)
                ax.vlines(draws.mean(), yi - 0.32, yi + 0.32, color=c["random"], linewidth=2)

            if (feature, cls) in rates.index:
                v, n = rates.loc[(feature, cls)]
                ax.hlines(yi, 0, v, color=c["selected"], linewidth=2)
                ax.plot(v, yi, "o", color=c["selected"], markersize=8)
                ax.annotate(f"{int(round(v * n))}/{int(n)}", (v, yi), textcoords="offset points",
                            xytext=(7, 0), va="center", fontsize=8.5, color=c["ink"])

        ax.set_title(f"{strength:g}× strength", fontsize=11, color=c["ink"], loc="left")
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, 0.5, 1])
        _percent_axis(ax.xaxis)
        _style_intervention_axes(ax, grid_axis="x")

    axes[0, 0].set_yticks(y, labels, fontsize=9, color=c["ink"])
    fig.supxlabel("Misclassified samples of the feature's class flipped to correct",
                  fontsize=9, color=c["ink_2"], y=0.08)

    handles = [plt.Line2D([], [], color=c["selected"], marker="o", linestyle="", markersize=8,
                          label="Selected feature (flipped / samples of its class)")]
    if n_draws:
        handles = [
            Patch(facecolor=c["random"], alpha=0.3, linewidth=0,
                  label=f"Random single features, same class (lowest to highest of {n_draws})"),
            plt.Line2D([], [], color=c["random"], linewidth=2,
                       label="Random single features, same class (mean)"),
        ] + handles
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Each selected feature on its own", fontsize=12, color=c["ink"],
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_intervention_per_sample(results, strength=1.0, output_path=None):
    """Each misclassified sample's probability of its correct class: before,
    after random feature sets (mean over draws), and after its true-class
    feature set, at one strength."""
    c = _INTERVENTION_COLOURS
    cond = INTERVENTION_CONDITIONS

    d = results[np.isclose(results["strength"], strength)].copy()
    correct_prob = lambda p, t: np.where(t == 1, p, 1 - p)
    d["p_before"] = correct_prob(d["prob_original"], d["target"])
    d["p_after"] = correct_prob(d["prob_new"], d["target"])

    sel = d[d["condition"] == cond["set"]].drop_duplicates("row").set_index("row")
    if sel.empty:
        print("No feature-set results to plot.")
        return None
    rnd = d[d["condition"] == cond["set_random"]].groupby("row")["p_after"].mean()

    m = sel[["class", "p_before", "p_after"]].join(rnd.rename("p_random"))
    m = m.sort_values(["class", "p_before"])

    fig, ax = plt.subplots(figsize=(8, 0.28 * len(m) + 1.8))
    y = np.arange(len(m))

    ax.axvline(0.5, color=c["ink_2"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.5, len(m) - 0.3, "decision boundary", ha="center", va="bottom",
            fontsize=8, color=c["ink_2"])

    ax.hlines(y, m["p_before"], m["p_after"], color=c["selected"], linewidth=1.5, alpha=0.6)
    ax.plot(m["p_before"], y, "o", markersize=8, markerfacecolor="white",
            markeredgecolor=c["ink_2"], markeredgewidth=1.5, label="Before (misclassified)")
    if m["p_random"].notna().any():
        ax.plot(m["p_random"], y, "o", markersize=8, color=c["random"],
                label="After random sets (mean)")
    ax.plot(m["p_after"], y, "o", markersize=8, color=c["selected"],
            label="After true-class features")
    for line in ax.lines:
        line.set_clip_on(False)  # keep markers at 0% / 100% whole

    classes = m["class"].to_numpy()
    for cls in pd.unique(classes):
        rows = np.flatnonzero(classes == cls)
        ax.text(-0.02, rows.mean(), f"truly {cls}\n(n={len(rows)})", ha="right", va="center",
                fontsize=9, color=c["ink"], transform=ax.get_yaxis_transform())
        if rows.min() > 0:
            ax.axhline(rows.min() - 0.5, color=c["grid"], linewidth=1)

    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(m) + 0.3)
    _percent_axis(ax.xaxis)
    ax.set_xlabel("Classifier's probability of the correct class", fontsize=9, color=c["ink_2"])
    _style_intervention_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
    ax.set_title(f"Each misclassified sample, all true-class features at {strength:g}× strength",
                 fontsize=12, color=c["ink"], loc="left")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig
