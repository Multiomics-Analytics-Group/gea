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

def differential_feature_activation(graph_acts_df, group_a, group_b, method="mannwhitney",
                                     padj_threshold=0.05, lfc_threshold=1.0):
    """
    Test each SAE feature for differential activation between two phenotype groups.

    Uses Benjamini-Hochberg FDR correction across all features.
    log2fc is defined as log2(mean_a / mean_b); positive values mean higher in group_a.

    A feature is marked `significant` only when it meets BOTH criteria:
    - adjusted p-value < padj_threshold  (statistical significance)
    - |log2fc| >= lfc_threshold           (practical/effect-size significance)

    Requiring both avoids the common trap where near-zero activations produce tiny
    p-values via a large relative fold-change that is biologically meaningless.

    Parameters
    ----------
    graph_acts_df : pd.DataFrame
        Output of extract_graph_activations (after filter_dead_features).
        Must have a 'label' column.
    group_a, group_b : str
        Phenotype labels to compare.
    method : str
        'mannwhitney' (non-parametric, default) or 'ttest'.
    padj_threshold : float
        BH-adjusted p-value cutoff (default 0.05).
    lfc_threshold : float
        Minimum absolute log2 fold-change to call a feature significant
        (default 1.0, i.e. 2-fold change). Set to 0 to use only p-value.

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
    df["significant"] = (df["p_adjusted"] < padj_threshold) & (df["log2fc"].abs() >= lfc_threshold)
    return df.sort_values("p_adjusted").reset_index(drop=True)


def volcano_plot(dfa_df, group_a, group_b, lfc_threshold=1.0,
                 padj_threshold=0.05, top_n=15, feature_labels=None, ax=None):
    """
    Volcano plot of differential SAE feature activations.

    Points in red are higher in group_a; in blue higher in group_b.
    Top-N most significant features are labelled.

    Parameters
    ----------
    feature_labels : dict, optional
        {feature_col: label_string} — if provided, labels replace raw feature
        names in point annotations. Useful for showing top gene symbols.
        Produced by ``label_features_by_genes``.

    Returns
    -------
    (fig, ax)
    """
    df = dfa_df.copy()
    df["neg_log10_padj"] = -np.log10(df["p_adjusted"].clip(lower=1e-300))

    is_up_a = (df["log2fc"] > lfc_threshold) & (df["p_adjusted"] < padj_threshold)
    is_up_b = (df["log2fc"] < -lfc_threshold) & (df["p_adjusted"] < padj_threshold)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.get_figure()

    ax.scatter(df.loc[~is_up_a & ~is_up_b, "log2fc"],
               df.loc[~is_up_a & ~is_up_b, "neg_log10_padj"],
               c="grey", alpha=0.4, s=20, label="n.s.")
    ax.scatter(df.loc[is_up_a, "log2fc"], df.loc[is_up_a, "neg_log10_padj"],
               c="tomato", alpha=0.7, s=28, label=f"Higher in {group_a}")
    ax.scatter(df.loc[is_up_b, "log2fc"], df.loc[is_up_b, "neg_log10_padj"],
               c="steelblue", alpha=0.7, s=28, label=f"Higher in {group_b}")

    top = df[is_up_a | is_up_b].nsmallest(top_n, "p_adjusted")
    texts = []
    for _, row in top.iterrows():
        label = (feature_labels.get(row["feature"], row["feature"])
                 if feature_labels else row["feature"])
        texts.append(ax.text(row["log2fc"], row["neg_log10_padj"], label,
                             fontsize=9, alpha=0.9))

    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.7),
                    expand=(1.2, 1.4), force_text=(0.5, 0.8))
    except ImportError:
        pass  # install adjustText for non-overlapping labels: pip install adjustText

    ax.axvline(lfc_threshold, ls="--", c="black", alpha=0.3, lw=1)
    ax.axvline(-lfc_threshold, ls="--", c="black", alpha=0.3, lw=1)
    ax.axhline(-np.log10(padj_threshold), ls="--", c="black", alpha=0.3, lw=1)
    ax.set_xlabel(f"log₂ fold change ({group_a} / {group_b})", fontsize=12)
    ax.set_ylabel("−log₁₀(adjusted p-value)", fontsize=12)
    ax.set_title(f"Differential Feature Activation: {group_a} vs {group_b}", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(framealpha=0.7, fontsize=10)
    fig.tight_layout()
    return fig, ax


# ── Dead Feature Filtering ────────────────────────────────────────────────────

def filter_dead_features(graph_acts_df, min_activation_frac=0.05):
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
                          gene_names=None, ensembl_to_symbol=None,
                          top_k=20, top_n_nodes=25,
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
    # Use only the most-activated graph's edges for subgraph visualization.
    # Stacking across top-k graphs is not safe: different graphs can have
    # different edge counts because LIONESS thresholds edges per sample.
    ref_idx = int(top_graph_idx[0])
    edge_scores = edge_acts_list[ref_idx].max(axis=1)  # [n_edges_ref]
    shared_edge_index = edge_indices_list[ref_idx]     # [2, n_edges_ref]

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
    def _node_label(i):
        raw = gene_names[i] if gene_names else str(i)
        if ensembl_to_symbol:
            return ensembl_to_symbol.get(raw, raw)
        return raw
    label_map = {i: _node_label(i) for i in range(n_nodes)}
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


# ── Gene Set Extraction & Enrichment ─────────────────────────────────────────

def get_attribution_gene_set(
    feature, sae_graph, gnn_model, data_loader,
    graph_acts_df, device, gene_names,
    ensembl_to_symbol=None, top_k=20, top_n=100,
):
    """
    Return the top-n genes most causally responsible for a graph SAE feature firing.

    Uses W_enc projection (contribution = z_node @ W_enc[:, f*] / N) averaged
    over the top-k most activated graphs. Positive score = gene pushes the
    embedding toward f* activating; higher magnitude = stronger causal role.

    Parameters
    ----------
    feature : int or str
    sae_graph : ShallowSAE
    gnn_model : GNNModel
    data_loader : DataLoader  (shuffle=False)
    graph_acts_df : pd.DataFrame
    device : torch.device
    gene_names : list of str
        Ordered list of Ensembl IDs matching graph node order.
    ensembl_to_symbol : dict, optional
        Ensembl ID → HGNC symbol mapping for labelling.
    top_k : int
        Number of most-activated graphs to average over.
    top_n : int
        Number of top genes to return.

    Returns
    -------
    pd.DataFrame
        Columns: gene_id, [symbol,] attribution_score. Sorted descending.
    """
    mean_attr, _, _ = attribute_nodes_to_graph_feature(
        feature=feature, sae_graph=sae_graph, gnn_model=gnn_model,
        data_loader=data_loader, graph_acts_df=graph_acts_df,
        device=device, top_k=top_k,
    )
    top_idx = np.argsort(mean_attr)[::-1][:top_n]
    gene_ids = [gene_names[i] for i in top_idx] if gene_names else [str(i) for i in top_idx]
    df = pd.DataFrame({"gene_id": gene_ids, "attribution_score": mean_attr[top_idx]})
    if ensembl_to_symbol is not None:
        df.insert(1, "symbol", df["gene_id"].map(ensembl_to_symbol).fillna(df["gene_id"]))
    return df


def get_concept_gene_set(
    feature, graph_acts_df, node_acts_list,
    node_concept, gene_names,
    ensembl_to_symbol=None, top_k=20, top_n=100,
):
    """
    Return the top-n genes by node SAE concept activation in graphs where f* fires.

    For each graph among the top-k most activated, the activation of a specified
    node SAE concept is extracted per gene. The mean across those graphs gives a
    consensus score: which genes most consistently exemplify this learned 'role'
    whenever feature f* is active.

    This is complementary to W_enc attribution:
    - Attribution answers "which genes drove f* to fire?"
    - Concept gene set answers "what shared biological role do those genes play,
      as learned by the node SAE?"

    Parameters
    ----------
    feature : int or str
    graph_acts_df : pd.DataFrame
    node_acts_list : list of np.array [n_nodes, n_node_features]
        Output of extract_node_activations.
    node_concept : int
        Index of the node SAE concept dimension to use (e.g. from get_top_node_concepts).
    gene_names : list of str
    ensembl_to_symbol : dict, optional
    top_k : int
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: gene_id, [symbol,] concept_score. Sorted descending.
    """
    feat_col = f"feature_{feature}" if isinstance(feature, int) else feature
    top_k_idx = np.argsort(graph_acts_df[feat_col].values)[::-1][:top_k]

    # [top_k, n_genes] → mean over graphs
    concept_scores = np.stack([
        node_acts_list[i][:, node_concept] for i in top_k_idx
    ]).mean(axis=0)

    top_idx = np.argsort(concept_scores)[::-1][:top_n]
    gene_ids = [gene_names[i] for i in top_idx] if gene_names else [str(i) for i in top_idx]
    df = pd.DataFrame({"gene_id": gene_ids, "concept_score": concept_scores[top_idx]})
    if ensembl_to_symbol is not None:
        df.insert(1, "symbol", df["gene_id"].map(ensembl_to_symbol).fillna(df["gene_id"]))
    return df


def run_enrichment(
    gene_symbols,
    gene_sets=None,
    organism="Human",
    padj_threshold=0.05,
):
    """
    Gene set overrepresentation analysis via gseapy.enrichr.

    Parameters
    ----------
    gene_symbols : list of str
        HGNC gene symbols to test. Unmapped Ensembl IDs are silently ignored
        by Enrichr, so filter them out before calling if possible.
    gene_sets : list of str, optional
        Enrichr library names. Defaults to KEGG, GO Biological Process, Reactome.
    organism : str
        Species string for Enrichr (default 'Human').
    padj_threshold : float
        Adjusted p-value cutoff.

    Returns
    -------
    pd.DataFrame
        Significant enriched terms sorted by adjusted p-value, with columns:
        Gene_set, Term, Overlap, P-value, Adjusted P-value, Genes.
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("Install gseapy: pip install gseapy")

    if gene_sets is None:
        gene_sets = [
            "KEGG_2021_Human",
            "GO_Biological_Process_2021",
            "Reactome_2022",
        ]

    res = gp.enrichr(
        gene_list=list(gene_symbols),
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
        cutoff=padj_threshold,
    )

    df = res.results.copy()
    sig = df[df["Adjusted P-value"] < padj_threshold].sort_values("Adjusted P-value").reset_index(drop=True)
    keep = [c for c in ["Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"] if c in sig.columns]
    return sig[keep]


def label_features_by_genes(
    features, sae_graph, gnn_model, data_loader, graph_acts_df,
    device, gene_names, ensembl_to_symbol=None, top_k=20, top_n=3,
):
    """
    For each SAE feature, return a short human-readable label made of the top-n
    attributed gene symbols. Designed to annotate volcano plot points.

    Parameters
    ----------
    features : list of str
        Feature column names (e.g. from dfa_df["feature"]).
    sae_graph, gnn_model, data_loader, graph_acts_df, device
        Same arguments as attribute_nodes_to_graph_feature.
    gene_names : list of str
        Ordered Ensembl IDs matching graph node order.
    ensembl_to_symbol : dict, optional
        Ensembl ID → HGNC symbol. If None, Ensembl IDs are used as labels.
    top_k : int
        Graphs to average attribution over.
    top_n : int
        Number of genes per label (default 3).

    Returns
    -------
    dict  { feature_col: "GENE1 / GENE2 / GENE3" }
    """
    labels = {}
    for feat in features:
        feat_col = f"feature_{feat}" if isinstance(feat, int) else feat
        mean_attr, _, _ = attribute_nodes_to_graph_feature(
            feature=feat_col, sae_graph=sae_graph, gnn_model=gnn_model,
            data_loader=data_loader, graph_acts_df=graph_acts_df,
            device=device, top_k=top_k,
        )
        top_idx = np.argsort(mean_attr)[::-1][:top_n]
        gene_ids = [gene_names[i] for i in top_idx] if gene_names else [str(i) for i in top_idx]
        syms = [ensembl_to_symbol.get(g, g) for g in gene_ids] if ensembl_to_symbol else gene_ids
        labels[feat_col] = " / ".join(syms)
    return labels


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

    # Edge scores from the most-activated graph only.
    # Stacking across top-k graphs would crash when edge counts differ per sample
    # (LIONESS thresholds edges independently for each sample).
    ref_idx = int(top_graph_idx[0])
    mean_edge_scores = edge_acts_list[ref_idx].max(axis=1)
    shared_edge_index = edge_indices_list[ref_idx]
    top_graph_labels = graph_acts_df["label"].values[top_graph_idx]

    return mean_node_scores, mean_edge_scores, shared_edge_index, top_graph_idx, top_graph_labels


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_feature_subgraph(node_scores, edge_scores, edge_index, gene_names=None,
                          ensembl_to_symbol=None,
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
    def _node_label(i):
        raw = gene_names[i] if gene_names else str(i)
        if ensembl_to_symbol:
            return ensembl_to_symbol.get(raw, raw)
        return raw
    label_map = {i: _node_label(i) for i in range(n_nodes)}

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
    g.ax_heatmap.set_xlabel("SAE features", fontsize=12)
    g.ax_heatmap.set_ylabel("Samples", fontsize=12)
    g.ax_heatmap.tick_params(axis="x", labelsize=10, rotation=90)

    handles = [plt.Rectangle((0, 0), 1, 1, color=label_colors[l]) for l in unique_labels]
    g.ax_col_dendrogram.legend(handles, unique_labels, loc="center", ncol=len(unique_labels),
                               title="Phenotype", framealpha=0.7)
    return g.fig, g.ax_heatmap


# ── SAE activations from exported embeddings ───────────────────────────────────

# Chart parameters. Magnitude gets one sequential hue (blue, light→dark); identity
# gets the validated categorical order. Text stays in ink colours, never a series
# colour. Swap these three lists to retheme every plot below.
_SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec",
             "#3987e5", "#256abf", "#184f95", "#0d366b"]
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_INK, _MUTED = "#52514e", "#898781"


def extract_sae_activations(sae_model, npz_file, device, batch_size=4096,
                            label_col="disease"):
    """
    Run a trained SAE over an exported embedding file and return its feature
    activations joined to the identity of every row.

    This is the ``EmbeddingDataset`` counterpart of ``extract_graph_activations``:
    the embeddings were already computed by ``gea.gea.export_embeddings``, so the
    GNN is not re-run, and it works for any level (graph, node or edge).

    Parameters
    ----------
    sae_model : ShallowSAE
        Trained SAE whose ``in_dim`` matches the embedding dimension of the file.
    npz_file : str or pathlib.Path
        File written by ``gea.gea.export_embeddings``.
    device : torch.device
    batch_size : int
        Rows per forward pass. Only affects memory, not results.
    label_col : str
        Identity column copied to a ``label`` column, which
        ``differential_feature_activation`` and ``plot_feature_activation_heatmap``
        expect. Defaults to the phenotype, ``disease``.

    Returns
    -------
    pd.DataFrame
        ``feature_0`` … ``feature_N`` plus every column of
        ``load_embedding_metadata`` (``graph_id``, ``gene`` / ``gene_a`` +
        ``gene_b``, ``disease``, ``cell_type``, the signed edge ``weight``, …) and
        ``label``. Row *i* is row *i* of the embedding file.

    Notes
    -----
    Memory scales as rows × SAE latent dim: harmless for the graph level (a few
    hundred rows) but a node- or edge-level file with ~10^6 rows and a 1024-unit
    SAE materialises billions of floats. Subsample those levels before calling.
    """
    from gea.dataloader import EmbeddingDataset, load_embedding_metadata

    dataset = EmbeddingDataset(npz_file)
    sae_model.to(device)
    sae_model.eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            z, _ = sae_model(dataset.embeddings[start:start + batch_size].to(device))
            chunks.append(z.cpu().numpy())

    acts = np.concatenate(chunks, axis=0)
    df = pd.DataFrame(acts, columns=[f"feature_{i}" for i in range(acts.shape[1])])
    df = pd.concat([df, load_embedding_metadata(npz_file)], axis=1)
    if label_col in df.columns:
        df["label"] = df[label_col]

    return df


def _annotation_colors(acts_df, annotate, palettes=None):
    """
    Assign a colour per category of each annotation column.

    Columns are served in the order given, so the annotation listed first — the
    one whose categories most need telling apart — gets the leading slots of the
    validated order, and slices do not overlap. Past eight categories in total the
    order restarts and a hue means one thing in one strip and something else in
    another; pass ``palettes`` if that repeat lands somewhere confusing.
    """
    if palettes is not None:
        return {col: dict(palettes[col]) for col in annotate}

    colors, offset = {}, 0
    for col in annotate:
        cats = sorted(acts_df[col].unique())
        colors[col] = {
            cat: _CATEGORICAL[(offset + i) % len(_CATEGORICAL)]
            for i, cat in enumerate(cats)
        }
        offset += len(cats)
    return {col: colors[col] for col in annotate}


def plot_sae_feature_clustermap(
    acts_df,
    annotate=("disease", "cell_type"),
    min_activation_frac=0.05,
    scale="feature",
    palettes=None,
    method="average",
    metric="euclidean",
    figsize=None,
    label_blocks=None,
    min_block=4,
    title="Graph-level SAE feature activations",
):
    """
    Clustered heatmap of SAE feature activations, with one categorical strip per
    biological annotation and dead features removed.

    Rows (samples) and columns (features) are both hierarchically clustered, so
    co-firing feature groups and the samples that share them fall next to each
    other. Cells carry magnitude on a single sequential hue; the annotation strips
    carry identity on the categorical order.

    Parameters
    ----------
    acts_df : pd.DataFrame
        Output of ``extract_sae_activations`` (or ``extract_graph_activations``)
        for the graph level: one row per graph.
    annotate : tuple of str
        Identity columns to draw as row strips, outermost first. Each gets its own
        legend, and the first listed gets the most separable hues.
    min_activation_frac : float
        Features active (> 0) in fewer than this fraction of rows are dropped, as
        are features with zero variance: neither can contribute to the clustering,
        and both flatten the colour scale.
    scale : {"feature", "none"}
        ``"feature"`` (default) min-max scales each feature column to 0–1 before
        clustering and drawing. SAE activation magnitudes differ by an order of
        magnitude between features, so on a raw scale two loud features saturate
        the ramp and every other column reads as empty. Scaling shows *which
        samples* fire a feature, which is what the clustering is about. Use
        ``"none"`` when absolute activation is the point.
    palettes : dict, optional
        ``{annotation_column: {category: colour}}`` to override the default hues.
    method, metric : str
        Linkage and distance passed to ``scipy``. ``euclidean`` is the default
        rather than ``correlation`` because an all-zero row makes correlation
        distance undefined.
    figsize : tuple, optional
        Defaults to a width that scales with the number of surviving features.
    label_blocks : str or None
        Annotation column whose contiguous blocks of clustered rows are labelled
        in text on the right edge, so identity does not rest on colour alone.
        Defaults to the annotation with the most categories; pass ``None`` to
        switch off.
    min_block : int
        Shortest run of consecutive same-category rows that earns a text label.
    title : str

    Returns
    -------
    g : seaborn.matrix.ClusterGrid
        ``g.figure`` for saving; ``g.dendrogram_row.reordered_ind`` for the row
        order actually drawn.
    alive : list of str
        Surviving feature columns, in input order.

    Notes
    -----
    A binary strip is safe: blue vs orange measures ΔE 24.7 under simulated
    protanopia (target 8) and 33.6 with normal vision (floor 15). A 7-category
    strip is not, and no choice of hues fixes it — every 7-hue subset of the
    palette was measured, and the best (the default here) still bottoms out at
    ΔE 13.2 with normal vision on its closest pair. Colour therefore cannot be
    the only channel: the text block labels, the per-strip legends and
    ``acts_df`` as the table view are what carry identity when two hues are
    hard to tell apart.
    """
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    annotate = [c for c in annotate if c in acts_df.columns]
    if not annotate:
        raise ValueError(
            "none of the requested annotation columns are present; "
            f"available: {[c for c in acts_df.columns if not c.startswith('feature_')]}"
        )

    feat_cols = [c for c in acts_df.columns if c.startswith("feature_")]
    alive = [
        c for c in feat_cols
        if (acts_df[c] > 0).mean() >= min_activation_frac and acts_df[c].std() > 0
    ]
    if not alive:
        raise ValueError(
            f"no feature is active in >= {min_activation_frac:.1%} of rows — the "
            "SAE may be collapsed, or min_activation_frac is too strict."
        )
    print(f"Alive: {len(alive)}/{len(feat_cols)} features "
          f"({len(feat_cols) - len(alive)} dead or constant, dropped)")

    matrix = acts_df[alive]
    colors = _annotation_colors(acts_df, annotate, palettes)
    row_colors = pd.DataFrame(
        {col: acts_df[col].map(colors[col]) for col in annotate},
        index=acts_df.index,
    )[list(annotate)]

    figsize = figsize or (min(20.0, 8.0 + 0.05 * len(alive)), 9.0)
    g = sns.clustermap(
        matrix,
        row_colors=row_colors,
        row_cluster=True,
        col_cluster=True,
        method=method,
        metric=metric,
        cmap=LinearSegmentedColormap.from_list("gea_sequential", _SEQ_BLUE),
        figsize=figsize,
        xticklabels=[c.replace("feature_", "") for c in alive] if len(alive) <= 60 else False,
        yticklabels=False,
        standard_scale=1 if scale == "feature" else None,
        dendrogram_ratio=(0.10, 0.13),
        colors_ratio=(0.022, 0.02),
        cbar_pos=(0.02, 0.78, 0.022, 0.15),
    )

    # Colourbar caption sits above the bar: as a rotated axis label it would run
    # into the row dendrogram and the annotation strips.
    g.ax_cbar.set_title(
        "activation\n(per feature)" if scale == "feature" else "activation",
        fontsize=8, color=_INK, loc="left", pad=6,
    )
    g.ax_cbar.tick_params(labelsize=7, colors=_MUTED, length=2)

    g.ax_heatmap.set_xlabel(f"SAE features ({len(alive)} alive)", color=_INK)
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="x", labelsize=7, colors=_MUTED, rotation=90)
    # The legends name the strips, so their axis labels would only collide
    g.ax_row_colors.set_xticks([])
    for spine in g.ax_heatmap.spines.values():
        spine.set_visible(False)
    if title:
        g.figure.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=13, color="#0b0b0b")

    # Text identity for the widest annotation: label each contiguous block of
    # clustered rows, so the strips are readable without discriminating hues.
    if label_blocks is None and annotate:
        label_blocks = max(annotate, key=lambda c: acts_df[c].nunique())
    if label_blocks:
        ordered = acts_df[label_blocks].values[g.dendrogram_row.reordered_ind]
        n_rows = len(ordered)
        ticks, labels, start = [], [], 0
        for i in range(1, n_rows + 1):
            if i == n_rows or ordered[i] != ordered[start]:
                if i - start >= min_block:
                    ticks.append((start + i) / 2)
                    labels.append(str(ordered[start]))
                start = i
        # Heatmap rows run top-to-bottom, y increasing downward
        g.ax_heatmap.set_yticks(ticks)
        g.ax_heatmap.set_yticklabels(labels, fontsize=7, color=_INK)
        g.ax_heatmap.tick_params(axis="y", length=2, colors=_MUTED)
        g.ax_heatmap.yaxis.set_label_position("right")

    # Squeeze every panel toward the left by the same factor — scaling x0 and
    # width together keeps the strips aligned with the heatmap rows — so the
    # legends have room on the right instead of overflowing the figure.
    for ax in (g.ax_heatmap, g.ax_row_dendrogram, g.ax_col_dendrogram,
               g.ax_row_colors, g.ax_cbar):
        if ax is None:
            continue
        box = ax.get_position()
        ax.set_position([box.x0 * 0.72, box.y0, box.width * 0.72, box.height])

    # One legend per strip, stacked to the right of the heatmap
    y = 0.97
    for col in annotate:
        handles = [Patch(facecolor=c, label=str(k)) for k, c in colors[col].items()]
        legend = g.figure.legend(
            handles=handles,
            title=col.replace("_", " "),
            loc="upper left",
            bbox_to_anchor=(0.85, y),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            handlelength=1.0,
            handleheight=1.0,
            labelcolor=_INK,
        )
        legend.get_title().set_color(_INK)
        y -= 0.055 * (len(handles) + 1.6)

    return g, alive
