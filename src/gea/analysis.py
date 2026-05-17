import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ── Activation Extraction ──────────────────────────────────────────────────────

def extract_graph_activations(sae_graph, gnn_model, data_loader, device, label_names=None):
    """
    Extract graph-level SAE feature activations for all graphs.

    Parameters
    ----------
    sae_graph : ShallowSAE
    gnn_model : GNNModel
    data_loader : DataLoader
        Must use shuffle=False so row order is deterministic.
    device : torch.device
    label_names : list, optional
        Maps integer labels to strings.

    Returns
    -------
    pd.DataFrame
        Rows = graphs. Columns: 'label', 'sample_name' (if available),
        'feature_0' … 'feature_N'.
    """
    sae_graph.eval()
    gnn_model.eval()

    all_acts, all_labels, all_names = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            _, z_graph = gnn_model.encode(batch)
            z_sae, _ = sae_graph(z_graph)
            all_acts.append(z_sae.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy().tolist())
            if hasattr(batch, "sample_name"):
                all_names.extend(batch.sample_name)

    acts = np.concatenate(all_acts, axis=0)  # [n_graphs, n_features]
    feat_cols = [f"feature_{i}" for i in range(acts.shape[1])]
    df = pd.DataFrame(acts, columns=feat_cols)

    df["label"] = [label_names[l] for l in all_labels] if label_names else all_labels
    if all_names:
        df["sample_name"] = all_names

    return df


def extract_node_activations(sae_node, gnn_model, data_loader, device):
    """
    Extract node-level SAE feature activations for every graph.

    Returns
    -------
    node_acts : list of np.array, shape [n_nodes, n_node_features] per graph
    labels    : list of int, one per graph
    """
    sae_node.eval()
    gnn_model.eval()

    node_acts, labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            z_node, _ = gnn_model.encode(batch)
            z_sae, _ = sae_node(z_node)
            for i in range(batch.num_graphs):
                s, e = batch.ptr[i].item(), batch.ptr[i + 1].item()
                node_acts.append(z_sae[s:e].cpu().numpy())
                labels.append(batch.y[i].item())

    return node_acts, labels


def extract_edge_activations(sae_edge, gnn_model, data_loader, device):
    """
    Extract edge-level SAE feature activations for every graph.

    Edge embeddings are computed as mean(z_node[src], z_node[dst]),
    matching the training procedure in train_sae_edge.

    Returns
    -------
    edge_acts    : list of np.array [n_edges, n_edge_features] per graph
    edge_indices : list of np.array [2, n_edges] local node indices per graph
    edge_weights : list of np.array [n_edges] original co-expression weights
    labels       : list of int
    """
    sae_edge.eval()
    gnn_model.eval()

    edge_acts, edge_indices, edge_weights, labels = [], [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            z_node, _ = gnn_model.encode(batch)

            ei = batch.edge_index
            src, dst = ei[0], ei[1]
            z_edge = (z_node[src] + z_node[dst]) / 2.0
            z_sae, _ = sae_edge(z_edge)

            edge_batch = batch.batch[src]
            for i in range(batch.num_graphs):
                mask = edge_batch == i
                offset = batch.ptr[i].item()
                edge_acts.append(z_sae[mask].cpu().numpy())
                edge_indices.append((ei[:, mask] - offset).cpu().numpy())
                edge_weights.append(batch.edge_attr[mask].cpu().numpy())
                labels.append(batch.y[i].item())

    return edge_acts, edge_indices, edge_weights, labels


# ── Differential Feature Activation ───────────────────────────────────────────

def differential_feature_activation(graph_acts_df, group_a, group_b, method="mannwhitney"):
    """
    Test each SAE feature for differential activation between two phenotype groups.

    Uses Benjamini-Hochberg FDR correction across all features.
    log2fc is defined as log2(mean_a / mean_b); positive values mean higher in group_a.

    Parameters
    ----------
    graph_acts_df : pd.DataFrame
        Output of extract_graph_activations. Must have a 'label' column.
    group_a, group_b : str
        Phenotype labels to compare.
    method : str
        'mannwhitney' (non-parametric, default) or 'ttest'.

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_a, mean_b, log2fc, stat, p_value, p_adjusted, significant.
        Sorted by p_adjusted ascending.
    """
    feat_cols = [c for c in graph_acts_df.columns if c.startswith("feature_")]
    a = graph_acts_df[graph_acts_df["label"] == group_a][feat_cols].values
    b = graph_acts_df[graph_acts_df["label"] == group_b][feat_cols].values

    rows = []
    for j, feat in enumerate(feat_cols):
        a_vals, b_vals = a[:, j], b[:, j]
        if method == "mannwhitney":
            stat, pval = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        else:
            stat, pval = stats.ttest_ind(a_vals, b_vals)
        mean_a, mean_b = float(a_vals.mean()), float(b_vals.mean())
        lfc = float(np.log2((mean_a + 1e-8) / (mean_b + 1e-8)))
        rows.append({"feature": feat, "mean_a": mean_a, "mean_b": mean_b,
                     "log2fc": lfc, "stat": float(stat), "p_value": float(pval)})

    df = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["p_adjusted"] = padj
    df["significant"] = df["p_adjusted"] < 0.05
    return df.sort_values("p_adjusted").reset_index(drop=True)


def volcano_plot(dfa_df, group_a, group_b, lfc_threshold=1.0,
                 padj_threshold=0.05, top_n=15, ax=None):
    """
    Volcano plot of differential SAE feature activations.

    Points in red are higher in group_a; in blue higher in group_b.
    Top-N most significant features are labelled.

    Returns
    -------
    (fig, ax)
    """
    df = dfa_df.copy()
    df["neg_log10_padj"] = -np.log10(df["p_adjusted"].clip(lower=1e-300))

    is_up_a = (df["log2fc"] > lfc_threshold) & (df["p_adjusted"] < padj_threshold)
    is_up_b = (df["log2fc"] < -lfc_threshold) & (df["p_adjusted"] < padj_threshold)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.scatter(df.loc[~is_up_a & ~is_up_b, "log2fc"],
               df.loc[~is_up_a & ~is_up_b, "neg_log10_padj"],
               c="grey", alpha=0.4, s=18, label="n.s.")
    ax.scatter(df.loc[is_up_a, "log2fc"], df.loc[is_up_a, "neg_log10_padj"],
               c="tomato", alpha=0.7, s=25, label=f"Higher in {group_a}")
    ax.scatter(df.loc[is_up_b, "log2fc"], df.loc[is_up_b, "neg_log10_padj"],
               c="steelblue", alpha=0.7, s=25, label=f"Higher in {group_b}")

    top = df[is_up_a | is_up_b].nsmallest(top_n, "p_adjusted")
    for _, row in top.iterrows():
        ax.annotate(row["feature"], (row["log2fc"], row["neg_log10_padj"]),
                    fontsize=7, alpha=0.85, xytext=(3, 3), textcoords="offset points")

    ax.axvline(lfc_threshold, ls="--", c="black", alpha=0.3, lw=1)
    ax.axvline(-lfc_threshold, ls="--", c="black", alpha=0.3, lw=1)
    ax.axhline(-np.log10(padj_threshold), ls="--", c="black", alpha=0.3, lw=1)
    ax.set_xlabel(f"log₂ fold change ({group_a} / {group_b})")
    ax.set_ylabel("−log₁₀(adjusted p-value)")
    ax.set_title(f"Differential Feature Activation: {group_a} vs {group_b}")
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    return fig, ax


# ── Dead Feature Filtering ────────────────────────────────────────────────────

def filter_dead_features(graph_acts_df, min_activation_frac=0.01):
    """
    Remove SAE features that are active in fewer than min_activation_frac of graphs.

    Dead features (activation == 0 on almost all samples) inflate the multiple-
    testing burden in DFA without contributing signal.

    Parameters
    ----------
    graph_acts_df : pd.DataFrame
        Output of extract_graph_activations.
    min_activation_frac : float
        Minimum fraction of graphs where a feature must be > 0 to be kept.

    Returns
    -------
    filtered_df : pd.DataFrame
        Same as input but feature columns restricted to alive features.
    alive_cols : list of str
        Names of the surviving feature columns.
    """
    feat_cols = [c for c in graph_acts_df.columns if c.startswith("feature_")]
    alive = [c for c in feat_cols if (graph_acts_df[c] > 0).mean() >= min_activation_frac]
    dead_n = len(feat_cols) - len(alive)
    print(f"Alive: {len(alive)}/{len(feat_cols)} features "
          f"({dead_n} dead, active in <{min_activation_frac*100:.1f}% of graphs)")
    meta_cols = [c for c in graph_acts_df.columns if not c.startswith("feature_")]
    return graph_acts_df[alive + meta_cols], alive


# ── Graph-feature → Node Attribution (W_enc projection) ───────────────────────

def attribute_nodes_to_graph_feature(feature, sae_graph, gnn_model, data_loader,
                                     graph_acts_df, device, top_k=20):
    """
    Compute each node's causal contribution to a graph-level SAE feature firing.

    Because z_graph = mean_pool(z_node), the pre-activation of feature f* is:

        pre_act(f*) = (z_graph - b_dec) @ W_enc[:, f*] + b_enc[f*]
                    = (1/N) * sum_i( z_node[i] @ W_enc[:, f*] ) + constants

    The per-node attribution is therefore z_node[i] @ W_enc[:, f*] / N.
    Positive values mean the node pushes the graph embedding toward f* activating;
    negative values push against it.

    Scores are averaged over the top-k most-activated graphs so the result
    represents the consensus attribution across the strongest examples.

    Parameters
    ----------
    feature : int or str
        Feature index (int) or column name (str, e.g. 'feature_42').
    sae_graph : ShallowSAE
    gnn_model : GNNModel
    data_loader : DataLoader
        Must use shuffle=False (same order as graph_acts_df).
    graph_acts_df : pd.DataFrame
        Output of extract_graph_activations.
    device : torch.device
    top_k : int

    Returns
    -------
    mean_attribution : np.array [n_nodes]
        Mean per-node attribution across top-k graphs.
    top_graph_idx : np.array [top_k]
    top_graph_labels : np.array [top_k]
    """
    feat_col = f"feature_{feature}" if isinstance(feature, int) else feature
    feat_idx = int(feat_col.split("_")[1])

    w_enc_f = sae_graph.W_enc[:, feat_idx].detach().to(device)  # [d_z]

    top_graph_idx = np.argsort(graph_acts_df[feat_col].values)[::-1][:top_k]
    top_k_set = set(top_graph_idx.tolist())

    gnn_model.eval()
    attributions = {}
    graph_counter = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            z_node, _ = gnn_model.encode(batch)
            for i in range(batch.num_graphs):
                if graph_counter in top_k_set:
                    s, e = batch.ptr[i].item(), batch.ptr[i + 1].item()
                    z_i = z_node[s:e]                    # [n_nodes, d_z]
                    n = z_i.shape[0]
                    attr = (z_i @ w_enc_f) / n           # [n_nodes]
                    attributions[graph_counter] = attr.cpu().numpy()
                graph_counter += 1

    attr_matrix = np.stack([attributions[i] for i in top_graph_idx])  # [top_k, n_nodes]
    top_graph_labels = graph_acts_df["label"].values[top_graph_idx]
    return attr_matrix.mean(axis=0), top_graph_idx, top_graph_labels


# ── Node and Edge Concept Extraction ──────────────────────────────────────────

def get_top_node_concepts(feature, graph_acts_df, node_acts_list, top_k=20, top_n=10):
    """
    Identify the most active node-level SAE features (concepts) in graphs where
    graph feature f* fires most strongly.

    Activations are averaged first across nodes within each graph, then across
    the top-k graphs, yielding a per-node-concept mean activation that reflects
    what the GNN consistently encodes at the node level in those samples.

    Parameters
    ----------
    feature : int or str
    graph_acts_df : pd.DataFrame
    node_acts_list : list of np.array [n_nodes, n_node_features]
    top_k : int
    top_n : int
        Number of top concepts to return.

    Returns
    -------
    pd.DataFrame
        Columns: node_concept, mean_activation. Sorted descending.
    """
    feat_col = f"feature_{feature}" if isinstance(feature, int) else feature
    top_k_idx = np.argsort(graph_acts_df[feat_col].values)[::-1][:top_k]

    # Mean over nodes per graph, then mean over top-k graphs → [n_node_features]
    mean_acts = np.stack([node_acts_list[i].mean(axis=0) for i in top_k_idx]).mean(axis=0)

    top_idx = np.argsort(mean_acts)[::-1][:top_n]
    return pd.DataFrame({
        "node_concept": [f"node_concept_{j}" for j in top_idx],
        "mean_activation": mean_acts[top_idx],
    })


def get_top_edge_concepts(feature, graph_acts_df, edge_acts_list, top_k=20, top_n=10):
    """
    Identify the most active edge-level SAE features (relationship types) in graphs
    where graph feature f* fires most strongly.

    Activations are averaged first across edges within each graph, then across
    the top-k graphs, yielding a per-edge-concept mean activation that reflects
    what co-expression relationship types the GNN encodes in those samples.

    Parameters
    ----------
    feature : int or str
    graph_acts_df : pd.DataFrame
    edge_acts_list : list of np.array [n_edges, n_edge_features]
    top_k : int
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: edge_concept, mean_activation. Sorted descending.
    """
    feat_col = f"feature_{feature}" if isinstance(feature, int) else feature
    top_k_idx = np.argsort(graph_acts_df[feat_col].values)[::-1][:top_k]

    mean_acts = np.stack([edge_acts_list[i].mean(axis=0) for i in top_k_idx]).mean(axis=0)

    top_idx = np.argsort(mean_acts)[::-1][:top_n]
    return pd.DataFrame({
        "edge_concept": [f"edge_concept_{j}" for j in top_idx],
        "mean_activation": mean_acts[top_idx],
    })


# ── Feature Co-activation ──────────────────────────────────────────────────────

def feature_coactivation(graph_acts_df, feature_cols=None, figsize=(10, 8)):
    """
    Compute and visualise pairwise Pearson correlations between graph SAE feature
    activations across all samples.

    Highly correlated feature clusters suggest the SAE has decomposed a shared
    biological process into multiple co-firing features. These clusters are more
    interpretable as a unit than any single feature alone.

    Parameters
    ----------
    graph_acts_df : pd.DataFrame
        Output of extract_graph_activations (ideally after filter_dead_features).
    feature_cols : list of str, optional
        Subset of feature columns to include. Defaults to all 'feature_*' columns.
    figsize : tuple

    Returns
    -------
    corr_matrix : pd.DataFrame
        [n_features × n_features] Pearson correlation matrix.
    (fig, ax)
    """
    import seaborn as sns

    if feature_cols is None:
        feature_cols = [c for c in graph_acts_df.columns if c.startswith("feature_")]

    corr = graph_acts_df[feature_cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=figsize)
    g = sns.clustermap(
        corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        figsize=figsize, xticklabels=False, yticklabels=False,
    )
    g.ax_heatmap.set_title("Graph SAE Feature Co-activation", pad=12)
    return corr, g.fig, g.ax_heatmap


# ── Full Explainability Pipeline ───────────────────────────────────────────────

def explain_graph_feature(feature, sae_graph, sae_node, sae_edge, gnn_model,
                          data_loader, graph_acts_df, node_acts_list,
                          edge_acts_list, edge_indices_list, device,
                          gene_names=None, top_k=20, top_n_nodes=25,
                          top_n_edges=40, top_n_concepts=10, figsize=(18, 7)):
    """
    Full three-level GEA explainability pipeline for a single graph-level SAE feature.

    Level 1 — Graph SAE:  which nodes caused f* to fire?
        → W_enc projection: node attribution scores (causal, direct)

    Level 2 — Node SAE:   what kind of node/concept is present in those graphs?
        → top node-level SAE features active in top-k graphs (semantic, generalizable)

    Level 3 — Edge SAE:   what kind of relationship exists between those nodes?
        → top edge-level SAE features active in top-k graphs + subgraph edge coloring

    The combined figure has three panels:
        Left   — subgraph where node color = W_enc attribution (level 1),
                  edge color = mean edge SAE activation (level 3)
        Center — bar chart of top node SAE concepts (level 2)
        Right  — bar chart of top edge SAE concepts (level 3)

    Parameters
    ----------
    feature : int or str
    sae_graph, sae_node, sae_edge : ShallowSAE
    gnn_model : GNNModel
    data_loader : DataLoader  (shuffle=False)
    graph_acts_df : pd.DataFrame
    node_acts_list : list  (from extract_node_activations)
    edge_acts_list : list  (from extract_edge_activations)
    edge_indices_list : list  (from extract_edge_activations)
    device : torch.device
    gene_names : list of str, optional
    top_k : int
        Graphs to aggregate over.
    top_n_nodes, top_n_edges : int
        Number of nodes/edges shown in the subgraph.
    top_n_concepts : int
        Number of node/edge concepts shown in bar charts.
    figsize : tuple

    Returns
    -------
    dict with keys:
        'node_attribution'  : np.array [n_nodes]   (W_enc projection scores)
        'edge_scores'       : np.array [n_edges]    (mean edge SAE activation)
        'edge_index'        : np.array [2, n_edges]
        'node_concepts'     : pd.DataFrame
        'edge_concepts'     : pd.DataFrame
        'top_graph_idx'     : np.array [top_k]
        'top_graph_labels'  : np.array [top_k]
        'fig'               : matplotlib Figure
    """
    feat_col = f"feature_{feature}" if isinstance(feature, int) else feature

    # ── Level 1: W_enc node attribution ───────────────────────────────────────
    node_attr, top_graph_idx, top_graph_labels = attribute_nodes_to_graph_feature(
        feature=feat_col, sae_graph=sae_graph, gnn_model=gnn_model,
        data_loader=data_loader, graph_acts_df=graph_acts_df,
        device=device, top_k=top_k,
    )

    # ── Level 2: Node SAE concepts ────────────────────────────────────────────
    node_concepts = get_top_node_concepts(
        feature=feat_col, graph_acts_df=graph_acts_df,
        node_acts_list=node_acts_list, top_k=top_k, top_n=top_n_concepts,
    )

    # ── Level 3: Edge SAE scores + concepts ───────────────────────────────────
    edge_scores = np.stack([
        edge_acts_list[i].max(axis=1) for i in top_graph_idx
    ]).mean(axis=0)
    shared_edge_index = edge_indices_list[top_graph_idx[0]]

    edge_concepts = get_top_edge_concepts(
        feature=feat_col, graph_acts_df=graph_acts_df,
        edge_acts_list=edge_acts_list, top_k=top_k, top_n=top_n_concepts,
    )

    # ── Phenotype composition of top-k graphs ─────────────────────────────────
    label_counts = pd.Series(top_graph_labels).value_counts()
    label_str = ", ".join(f"{l}: {n}" for l, n in label_counts.items())

    # ── Combined figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.35)
    ax_graph = fig.add_subplot(gs[0])
    ax_node  = fig.add_subplot(gs[1])
    ax_edge  = fig.add_subplot(gs[2])

    # Panel 1: subgraph — node color = W_enc attribution, edge color = edge SAE
    n_nodes = len(node_attr)
    label_map = {i: (gene_names[i] if gene_names else str(i)) for i in range(n_nodes)}
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, score=float(node_attr[i]))

    top_edge_idx = np.argsort(edge_scores)[::-1][:top_n_edges]
    for idx in top_edge_idx:
        u, v = int(shared_edge_index[0, idx]), int(shared_edge_index[1, idx])
        G.add_edge(u, v, score=float(edge_scores[idx]))

    top_node_set = set(np.argsort(node_attr)[::-1][:top_n_nodes].tolist())
    edge_node_set = {int(shared_edge_index[0, i]) for i in top_edge_idx} | \
                    {int(shared_edge_index[1, i]) for i in top_edge_idx}
    subG = G.subgraph(top_node_set | edge_node_set).copy()
    pos = nx.spring_layout(subG, seed=42, k=2.5)

    node_list = list(subG.nodes())
    n_sc = np.array([subG.nodes[n]["score"] for n in node_list])
    n_sz = 200 + 800 * (n_sc / (n_sc.max() + 1e-8))
    edge_list = list(subG.edges())
    e_sc = np.array([subG[u][v]["score"] for u, v in edge_list]) if edge_list else np.array([])

    if edge_list:
        nx.draw_networkx_edges(subG, pos, ax=ax_graph, edgelist=edge_list,
                               edge_color=e_sc, edge_cmap=cm.Blues,
                               width=1.5 + 2.0 * (e_sc / (e_sc.max() + 1e-8)), alpha=0.6)
    nc = nx.draw_networkx_nodes(subG, pos, ax=ax_graph, nodelist=node_list,
                                node_color=n_sc, node_size=n_sz,
                                cmap=cm.RdBu_r, alpha=0.9)
    plt.colorbar(nc, ax=ax_graph, label="Node attribution (W_enc)", shrink=0.6)
    nx.draw_networkx_labels(subG, pos, labels={n: label_map[n] for n in node_list},
                            ax=ax_graph, font_size=7, font_weight="bold")
    ax_graph.set_title(
        f"{feat_col}  |  top-{top_k} graphs: {label_str}",
        fontsize=10, pad=8,
    )
    ax_graph.axis("off")

    # Panel 2: top node SAE concepts
    ax_node.barh(node_concepts["node_concept"][::-1],
                 node_concepts["mean_activation"][::-1],
                 color="steelblue", alpha=0.8)
    ax_node.set_xlabel("Mean activation")
    ax_node.set_title("Top node concepts\n(node SAE)", fontsize=10)
    ax_node.tick_params(axis="y", labelsize=8)

    # Panel 3: top edge SAE concepts
    ax_edge.barh(edge_concepts["edge_concept"][::-1],
                 edge_concepts["mean_activation"][::-1],
                 color="tomato", alpha=0.8)
    ax_edge.set_xlabel("Mean activation")
    ax_edge.set_title("Top edge concepts\n(edge SAE)", fontsize=10)
    ax_edge.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"GEA Explainability Pipeline — {feat_col}", fontsize=13, y=1.01)
    fig.tight_layout()

    return {
        "node_attribution": node_attr,
        "edge_scores": edge_scores,
        "edge_index": shared_edge_index,
        "node_concepts": node_concepts,
        "edge_concepts": edge_concepts,
        "top_graph_idx": top_graph_idx,
        "top_graph_labels": top_graph_labels,
        "fig": fig,
    }


# ── Subgraph Tracing ───────────────────────────────────────────────────────────

def trace_feature_to_subgraph(feature, graph_acts_df, node_acts_list,
                               edge_acts_list, edge_indices_list,
                               top_k=20, node_agg="mean"):
    """
    For a graph-level SAE feature, find the top-k most activated graphs and
    aggregate node/edge SAE activations across them to identify a consensus subgraph.

    Because all graphs share the same PPI topology (only weights differ),
    edge indices are taken from the first graph in the top-k set and are
    assumed to be consistent across the dataset.

    Parameters
    ----------
    feature : int or str
        Feature index (int) or column name (str, e.g. 'feature_42').
    graph_acts_df : pd.DataFrame
        Output of extract_graph_activations. Must have a 'label' column.
    node_acts_list : list
        Output of extract_node_activations.
    edge_acts_list : list
        Output of extract_edge_activations.
    edge_indices_list : list
        Local edge indices per graph from extract_edge_activations.
    top_k : int
        Number of top-activated graphs to aggregate over.
    node_agg : str
        How to collapse the node SAE feature dimension: 'mean' or 'max'.

    Returns
    -------
    mean_node_scores : np.array [n_nodes]
        Per-node mean activation across top-k graphs.
    mean_edge_scores : np.array [n_edges]
        Per-edge mean activation (max over edge features) across top-k graphs.
    shared_edge_index : np.array [2, n_edges]
        Local edge indices (topology shared across all graphs).
    top_graph_idx : np.array [top_k]
        Indices into graph_acts_df of the top-k graphs.
    top_graph_labels : np.array [top_k]
        Phenotype labels of the top-k graphs.
    """
    feat_col = f"feature_{feature}" if isinstance(feature, int) else feature
    scores = graph_acts_df[feat_col].values
    top_graph_idx = np.argsort(scores)[::-1][:top_k]

    # Collapse SAE feature dim per node, then average across top-k graphs
    agg_fn = np.max if node_agg == "max" else np.mean
    node_scores = np.stack([agg_fn(node_acts_list[i], axis=1) for i in top_graph_idx])
    mean_node_scores = node_scores.mean(axis=0)

    # Max over edge features, average across top-k graphs
    edge_scores = np.stack([edge_acts_list[i].max(axis=1) for i in top_graph_idx])
    mean_edge_scores = edge_scores.mean(axis=0)

    shared_edge_index = edge_indices_list[top_graph_idx[0]]
    top_graph_labels = graph_acts_df["label"].values[top_graph_idx]

    return mean_node_scores, mean_edge_scores, shared_edge_index, top_graph_idx, top_graph_labels


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_feature_subgraph(node_scores, edge_scores, edge_index, gene_names=None,
                          top_n_nodes=25, top_n_edges=40, title="", figsize=(12, 10)):
    """
    Visualize the feature-associated subgraph.

    Nodes are sized and colored by mean SAE activation. Edges are colored
    by mean SAE activation weight. Only the top-N nodes/edges are shown
    for readability.

    Parameters
    ----------
    node_scores : np.array [n_nodes]
    edge_scores : np.array [n_edges]
    edge_index : np.array [2, n_edges]
        Local (per-graph) edge indices.
    gene_names : list of str, optional
        If provided, node labels are gene symbols.
    top_n_nodes : int
    top_n_edges : int
    title : str
    figsize : tuple

    Returns
    -------
    (fig, ax, subgraph)
        subgraph is a networkx.Graph of the visible nodes/edges.
    """
    n_nodes = len(node_scores)
    label_map = {i: (gene_names[i] if gene_names else str(i)) for i in range(n_nodes)}

    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, score=float(node_scores[i]))

    top_edge_idx = np.argsort(edge_scores)[::-1][:top_n_edges]
    for idx in top_edge_idx:
        u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
        G.add_edge(u, v, score=float(edge_scores[idx]))

    top_node_set = set(np.argsort(node_scores)[::-1][:top_n_nodes].tolist())
    edge_node_set = {int(edge_index[0, i]) for i in top_edge_idx} | \
                    {int(edge_index[1, i]) for i in top_edge_idx}
    subG = G.subgraph(top_node_set | edge_node_set).copy()

    pos = nx.spring_layout(subG, seed=42, k=2.5)

    node_list = list(subG.nodes())
    n_sc = np.array([subG.nodes[n]["score"] for n in node_list])
    n_sz = 200 + 800 * (n_sc / (n_sc.max() + 1e-8))

    edge_list = list(subG.edges())
    e_sc = np.array([subG[u][v]["score"] for u, v in edge_list]) if edge_list else np.array([])

    fig, ax = plt.subplots(figsize=figsize)

    if edge_list:
        nx.draw_networkx_edges(
            subG, pos, ax=ax, edgelist=edge_list,
            edge_color=e_sc, edge_cmap=cm.Blues,
            width=1.5 + 2.0 * (e_sc / (e_sc.max() + 1e-8)), alpha=0.6,
        )

    nc = nx.draw_networkx_nodes(
        subG, pos, ax=ax, nodelist=node_list,
        node_color=n_sc, node_size=n_sz, cmap=cm.YlOrRd, alpha=0.9,
    )
    plt.colorbar(nc, ax=ax, label="Mean node activation", shrink=0.7)
    nx.draw_networkx_labels(subG, pos, labels={n: label_map[n] for n in node_list},
                            ax=ax, font_size=7, font_weight="bold")

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax, subG


def plot_feature_activation_heatmap(graph_acts_df, features, label_col="label",
                                    figsize=None):
    """
    Heatmap of SAE feature activations, grouped by phenotype label.

    Parameters
    ----------
    graph_acts_df : pd.DataFrame
        Output of extract_graph_activations.
    features : list of str
        Feature column names to show (e.g. top DA features).
    label_col : str

    Returns
    -------
    (fig, ax)
    """
    import seaborn as sns

    df = graph_acts_df.sort_values(label_col)[features + [label_col]].copy()
    labels = df.pop(label_col)

    unique_labels = labels.unique()
    palette = sns.color_palette("tab10", len(unique_labels))
    label_colors = {lab: palette[i] for i, lab in enumerate(unique_labels)}
    row_colors = labels.map(label_colors)

    figsize = figsize or (max(8, len(features) * 0.5), max(6, len(df) * 0.04))
    g = sns.clustermap(
        df, row_colors=row_colors, col_cluster=True, row_cluster=False,
        cmap="viridis", figsize=figsize,
        yticklabels=False, xticklabels=True,
    )
    g.ax_heatmap.set_xlabel("SAE features")
    g.ax_heatmap.set_ylabel("Samples")

    handles = [plt.Rectangle((0, 0), 1, 1, color=label_colors[l]) for l in unique_labels]
    g.ax_col_dendrogram.legend(handles, unique_labels, loc="center", ncol=len(unique_labels),
                               title="Phenotype", framealpha=0.7)
    return g.fig, g.ax_heatmap
