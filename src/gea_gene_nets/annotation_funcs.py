"""
Annotation of SAE features learnt on gene co-expression network embeddings.

This is the gene-network counterpart of ``gea_molecules.annotation_funcs``. There,
a *concept* is a chemical motif and the annotation of an atom is "does this atom
belong to a match of this SMARTS pattern?" — a mask over the atoms of one molecule,
obtained with ``motifAnnotation`` and attached to every node embedding so the
framework in ``gea.analysis`` can score each SAE feature against every concept.

Here a *concept* is a **gene set** (GO term, KEGG / Reactome / WikiPathways
pathway) and the annotation of a node is "does this gene belong to this term?".
The mapping is one-to-one:

    molecules                        gene networks
    ---------------------------      ------------------------------------------
    SMILES                       →   graph_id (one pseudobulk profile = one graph)
    atom                         →   gene (node)
    motif dict {name: SMARTS}    →   gene-set library {term: [symbols]}
    motifAnnotation(mol, motifs) →   geneSetAnnotation(gene_list, gene_sets)
    F1(motif | feature)          →   F1(term | feature)

The one structural difference drives the whole design of this module: molecules
have different atoms, but **every graph here shares the same node universe** (the
PPI-filtered gene list). Term membership is therefore a property of the *gene*,
not of the graph, and it is the SAE *activations* that vary from graph to graph.
So an annotation is computed **per graph** and then integrated across graphs —
which is exactly what makes the result interesting: a term that a feature encodes
in every profile is a housekeeping concept, while one it only encodes in the T2D
beta-cell profiles is a phenotype-specific concept.

Pipeline (the steps commented in the original draft of this file)

1. ``fetch_gene_sets`` + ``geneSetAnnotation`` — build the term masks over the
   node universe. ``gene_set_matrix`` is the same thing as a genes × terms table.
2. ``annotate_node_features`` — for each graph, threshold that graph's node-level
   SAE activations and score every (feature, term) pair by F1, keeping the top-k
   terms per feature. Returns one tidy row per (graph, feature, term).
3. ``consensus_feature_terms`` / ``assign_feature_terms`` — integrate across
   graphs into one annotation per feature, with a confidence (support × F1).
4. ``group_feature_terms`` — the same integration within each phenotype or cell
   type, i.e. condition-specific annotations.
5. ``plot_feature_term_heatmap`` / ``plot_feature_annotation_bars`` /
   ``plot_group_term_dotplot`` — visualisation.
6. ``sae_health``, ``term_neighbourhood_enrichment`` and the UMAP views —
   diagnostics for when the F1 scores come out low, which separate the possible
   causes: the SAE is badly trained, the pathway signal is not in the embedding to
   begin with, or the features and the gene sets genuinely only partly coincide.
   See the notes on that section below.

``annotate_node_geneNets`` and ``annotate_edge_geneNets`` are the direct
analogues of ``annotate_node_GroverEmbeds``: they bake the term masks into a new
``.npz`` so the existing ``gea.analysis.gea_annotation`` machinery — written for
scalar per-row concepts — runs on gene networks unchanged.
"""

import numpy as np
import pandas as pd
import torch


# ── The gene-set universe ─────────────────────────────────────────────────────

def fetch_gene_sets(libraries=None, organism="Human", gmt_files=None):
    """
    Download gene-set libraries from Enrichr (and/or read local ``.gmt`` files).

    The gene-network analogue of ``scripts/gea_molecules/create_motif_dict.py``:
    it produces the *dictionary of concepts* that ``geneSetAnnotation`` then
    turns into masks over the node universe.

    Parameters
    ----------
    libraries : list of str, optional
        Enrichr library names (see ``gseapy.get_library_name(organism)``).
        Defaults to KEGG + WikiPathways + GO Biological Process for the requested
        organism.
    organism : str
        Enrichr organism: ``'Human'``, ``'Mouse'``, ``'Fly'``, ``'Yeast'``, …
        Mouse libraries are served by modEnrichr, so pass ``'Mouse'`` for the
        ``ENSMUSG`` cohorts rather than querying human pathways.
    gmt_files : list of str or pathlib.Path, optional
        Local ``.gmt`` files, read in addition to (or instead of) the Enrichr
        libraries. Useful offline, or for a curated set of terms.

    Returns
    -------
    dict
        ``{term: [gene symbols]}``. Terms are prefixed with their library
        (``"KEGG_2019_Mouse | Insulin secretion"``) so that a term name is unique
        and its provenance stays readable in every downstream table and plot.

    Notes
    -----
    Requires ``gseapy`` (``pip install gseapy``) and, for ``libraries``, network
    access to Enrichr.
    """
    if libraries is None:
        libraries = (
            ["KEGG_2019_Mouse", "WikiPathways_2019_Mouse", "GO_Biological_Process_2023"]
            if str(organism).lower().startswith("mouse")
            else ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023"]
        )

    gene_sets = {}

    if libraries:
        try:
            import gseapy as gp
        except ImportError:
            raise ImportError("Install gseapy to download Enrichr libraries: pip install gseapy")

        for name in libraries:
            library = gp.get_library(name=name, organism=organism)
            gene_sets.update({f"{name} | {term}": members for term, members in library.items()})
            print(f"{name}: {len(library)} terms")

    for path in gmt_files or []:
        with open(path) as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) > 2:
                    gene_sets[f"{path} | {fields[0]}"] = [g for g in fields[2:] if g]

    print(f"Total: {len(gene_sets)} terms")
    return gene_sets


def geneSetAnnotation(
    gene_list,
    gene_sets,
    ensembl_to_symbol=None,
    min_genes=5,
    max_genes=500,
):
    """
    Mask every gene of the node universe against every gene set.

    Direct analogue of ``gea_molecules.annotation_funcs.motifAnnotation``: that
    function returns one 0/1 mask per motif over the atoms of a molecule, this one
    returns one 0/1 mask per term over the genes of the graphs.

    Parameters
    ----------
    gene_list : list of str
        The node universe **in graph node order** — i.e. the ``gene_names`` passed
        to ``gea.gea.export_embeddings`` (``get_gene_list(filt_counts_ppi)``).
        Usually Ensembl IDs.
    gene_sets : dict
        ``{term: [gene symbols]}``, from ``fetch_gene_sets``.
    ensembl_to_symbol : dict, optional
        ``{Ensembl ID: gene symbol}``, as built with ``mygene`` in the tutorials.
        Required when ``gene_list`` holds Ensembl IDs, since Enrichr libraries are
        indexed by symbol. Matching is case-insensitive, so mouse symbols
        (``Ins1``) match the upper-cased Enrichr entries (``INS1``).
    min_genes, max_genes : int
        Keep only terms with between ``min_genes`` and ``max_genes`` members
        **present in the node universe**. Terms below the floor cannot yield a
        meaningful F1 (a single gene makes any feature that fires on it a perfect
        match), and terms above the ceiling are too unspecific to interpret.

    Returns
    -------
    dict
        ``{term: mask}`` where ``mask`` is a list of 0/1 of length
        ``len(gene_list)``, aligned to ``gene_list``.
    """
    symbols = [
        str(ensembl_to_symbol.get(g, g) if ensembl_to_symbol else g).upper()
        for g in gene_list
    ]

    gene_dict = {}

    for term, members in gene_sets.items():
        members = {str(m).upper() for m in members}
        mask = [1 if s in members else 0 for s in symbols]

        n_present = sum(mask)
        if n_present < min_genes or n_present > max_genes:
            continue

        gene_dict[term] = mask

    sizes = np.array([sum(m) for m in gene_dict.values()]) if gene_dict else np.array([0])
    print(
        f"{len(gene_dict)}/{len(gene_sets)} terms with {min_genes}–{max_genes} of the "
        f"{len(gene_list)} genes (median {np.median(sizes):.0f} genes/term)"
    )
    return gene_dict


def gene_set_matrix(term_masks, gene_list):
    """
    The term masks as a genes × terms table, for inspection and for passing to
    ``annotate_node_features`` without repeating ``gene_list``.

    Parameters
    ----------
    term_masks : dict
        Output of ``geneSetAnnotation``.
    gene_list : list of str
        Same gene order used to build the masks.

    Returns
    -------
    pd.DataFrame
        Boolean, indexed by gene, one column per term.
    """
    return pd.DataFrame(
        {term: np.asarray(mask, dtype=bool) for term, mask in term_masks.items()},
        index=list(gene_list),
    )


def _as_term_frame(term_masks, gene_list=None):
    """Accept either the ``geneSetAnnotation`` dict (+ gene order) or its table."""
    if isinstance(term_masks, pd.DataFrame):
        return term_masks.astype(bool)
    if gene_list is None:
        raise ValueError(
            "gene_list is required when term_masks is a dict of masks: the masks "
            "are positional, so the gene order they were built with has to be "
            "given. Pass gene_set_matrix(term_masks, gene_list) to skip this."
        )
    return gene_set_matrix(term_masks, gene_list)


# ── Bridge to the existing EmbeddingDataset / gea_annotation framework ────────

def annotate_node_geneNets(npz_file, term_masks, gene_list=None, out_file=None,
                           keep_annotations=()):
    """
    Rewrite a node-level embedding export with gene-set concepts as annotations.

    This is the counterpart of ``annotate_node_GroverEmbeds``: the embeddings are
    already computed (by ``gea.gea.export_embeddings``), so the only thing added is
    one scalar 0/1 annotation per term per row, which is the layout
    ``gea.analysis.gea_annotation`` / ``concept_comparison`` expect. The result can
    be read straight back with ``EmbeddingDataset`` and fed to the same
    F1-per-concept code used for the molecular motifs.

    Term membership depends only on the gene, so the per-row annotation dicts are
    **shared** between all rows of the same gene (1030 dicts rather than 200k).
    That keeps memory and the pickled file small, and is why row-varying
    annotations are dropped by default.

    Parameters
    ----------
    npz_file : str or pathlib.Path
        Node-level file written by ``export_embeddings`` (``*_node.npz``).
    term_masks : dict or pd.DataFrame
        Output of ``geneSetAnnotation`` (with ``gene_list``), or of
        ``gene_set_matrix``.
    gene_list : list of str, optional
        Gene order of the masks; not needed when ``term_masks`` is a table.
    out_file : str or pathlib.Path, optional
        When given, the result is written there with ``np.savez_compressed``.
    keep_annotations : iterable of str
        Row-varying annotations of the source file to carry over (e.g.
        ``("expression", "degree")``). Anything kept forces one dict per row, so
        only ask for what a concept analysis actually needs.

    Returns
    -------
    dict
        ``embeddings``, ``annotations``, ``entities``, ``prediction``, ``target``
        (plus the identity columns and ``term_names``) — the ``EmbeddingDataset``
        layout, same as the molecular annotators return.
    """
    terms = _as_term_frame(term_masks, gene_list)
    data = np.load(npz_file, allow_pickle=True)
    genes = data["gene"]

    missing = set(np.unique(genes)) - set(terms.index)
    if missing:
        raise ValueError(
            f"{len(missing)} genes of the export are absent from the term masks "
            f"(e.g. {sorted(missing)[:3]}). The masks must cover the node universe."
        )

    # One annotation dict per gene, referenced by every row of that gene
    per_gene = {
        gene: {term: float(value) for term, value in zip(terms.columns, row)}
        for gene, row in zip(terms.index, terms.to_numpy())
    }

    keep_annotations = list(keep_annotations)
    if keep_annotations:
        source = data["annotations"]
        annotations = [
            {**per_gene[gene], **{k: source[i][k] for k in keep_annotations}}
            for i, gene in enumerate(genes)
        ]
    else:
        annotations = [per_gene[gene] for gene in genes]

    result = {
        "embeddings": data["embeddings"],
        "annotations": np.array(annotations, dtype=object),
        "entities": data["entities"],
        "prediction": data["prediction"],
        "target": data["target"],
        "term_names": np.array(list(terms.columns), dtype=object),
    }
    result.update({
        col: data[col] for col in ("graph_id", "gene", "disease", "cell_type")
        if col in data.files
    })

    if out_file is not None:
        np.savez_compressed(out_file, **result)
        print(f"{len(annotations):,} rows × {terms.shape[1]} gene-set concepts → {out_file}")

    return result


def annotate_edge_geneNets(npz_file, term_masks, gene_list=None, out_file=None,
                           keep_annotations=("weight",)):
    """
    Same as ``annotate_node_geneNets`` for an edge-level export, where an edge is
    annotated with a term when **both** endpoints belong to it.

    A within-term edge is the co-expression relationship a pathway would predict,
    so an edge SAE feature that matches such a mask has learnt "this pair of genes
    is wired inside pathway X" rather than a property of a single gene. Unlike the
    node case this annotation does vary between graphs, because LIONESS thresholds
    each profile's edges independently.

    Parameters
    ----------
    npz_file : str or pathlib.Path
        Edge-level file written by ``export_embeddings`` (``*_edge.npz``).
    term_masks, gene_list, out_file
        See ``annotate_node_geneNets``.
    keep_annotations : iterable of str
        Row-varying annotations to carry over; the signed co-expression
        ``weight`` is kept by default since edge features are usually read
        together with it.

    Returns
    -------
    dict
        ``EmbeddingDataset`` layout, as above.
    """
    terms = _as_term_frame(term_masks, gene_list)
    data = np.load(npz_file, allow_pickle=True)
    gene_a, gene_b = data["gene_a"], data["gene_b"]

    membership = {gene: row for gene, row in zip(terms.index, terms.to_numpy())}
    term_names = list(terms.columns)
    source = data["annotations"]
    keep_annotations = list(keep_annotations)

    # Cached per gene pair: the PPI topology is shared, so the same pair recurs in
    # most graphs and its mask only has to be built once.
    pair_cache = {}
    annotations = []
    for i, (a, b) in enumerate(zip(gene_a, gene_b)):
        key = (a, b)
        if key not in pair_cache:
            both = membership[a] & membership[b]
            pair_cache[key] = {term: float(v) for term, v in zip(term_names, both)}
        entry = pair_cache[key]
        if keep_annotations:
            entry = {**entry, **{k: source[i][k] for k in keep_annotations}}
        annotations.append(entry)

    result = {
        "embeddings": data["embeddings"],
        "annotations": np.array(annotations, dtype=object),
        "entities": data["entities"],
        "prediction": data["prediction"],
        "target": data["target"],
        "term_names": np.array(term_names, dtype=object),
    }
    result.update({
        col: data[col] for col in ("graph_id", "gene_a", "gene_b", "disease", "cell_type")
        if col in data.files
    })

    if out_file is not None:
        np.savez_compressed(out_file, **result)
        print(f"{len(annotations):,} edges × {len(term_names)} gene-set concepts → {out_file}")

    return result


# ── F1 of every (feature, term) pair ─────────────────────────────────────────

def feature_maxima(sae_model, embeddings, device="cpu", batch_size=8192):
    """
    Per-feature activation maximum and firing frequency over a whole export.

    The ``EmbeddingDataset``-free equivalent of
    ``gea.analysis.compute_feature_maxima``. The maxima are computed **globally**,
    over every row of the file, so that a relative threshold means the same thing
    in every graph and F1 scores stay comparable across graphs.

    Parameters
    ----------
    sae_model : ShallowSAE
        Trained node-level SAE.
    embeddings : torch.Tensor
        ``[N, D]`` embedding matrix, e.g. ``EmbeddingDataset(npz).embeddings``.
    device : torch.device or str
    batch_size : int
        Rows per forward pass; affects memory only.

    Returns
    -------
    maxima : torch.Tensor ``[n_features]``
        Clamped away from zero so dividing by it is safe.
    frequency : torch.Tensor ``[n_features]``
        Fraction of rows where the feature is > 0.
    """
    sae_model.to(device)
    sae_model.eval()

    maxima = torch.zeros(sae_model.latent_dim, device=device)
    active = torch.zeros(sae_model.latent_dim, device=device)

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            z, _ = sae_model(embeddings[start:start + batch_size].to(device))
            maxima = torch.maximum(maxima, z.max(dim=0).values)
            active += (z > 0).sum(dim=0)

    maxima.clamp_(min=1e-8)
    return maxima, active / len(embeddings)


def feature_term_f1(binary_acts, term_matrix):
    """
    F1, precision and recall of every (feature, term) pair, in one matmul.

    Treats each SAE feature as a binary classifier of term membership over the
    nodes of one graph: a feature that fires on exactly the genes of a term gets
    F1 = 1. Same definition as ``gea.analysis.calculate_f1``, but scored for all
    features against all terms at once —

        tp = Aᵀ·T,  fp = |A| − tp,  fn = |T| − tp,  F1 = 2tp / (|A| + |T|)

    — which is what makes a per-graph sweep over hundreds of terms affordable.

    Parameters
    ----------
    binary_acts : torch.Tensor
        ``[n_nodes, n_features]`` bool/float — thresholded activations of the nodes
        of one graph (see ``gea.analysis.activations_preparation``).
    term_matrix : torch.Tensor
        ``[n_nodes, n_terms]`` bool/float — term membership of the same nodes,
        in the same row order.

    Returns
    -------
    dict of torch.Tensor, each ``[n_features, n_terms]``
        ``f1``, ``precision``, ``recall``, ``tp``; plus ``n_active``
        ``[n_features]`` and ``n_term_genes`` ``[n_terms]``.
    """
    acts = binary_acts.float()
    terms = term_matrix.float()

    tp = acts.T @ terms                       # [n_features, n_terms]
    n_active = acts.sum(dim=0)                # [n_features]
    n_term = terms.sum(dim=0)                 # [n_terms]

    return {
        "f1": 2 * tp / (n_active[:, None] + n_term[None, :] + 1e-8),
        "precision": tp / (n_active[:, None] + 1e-8),
        "recall": tp / (n_term[None, :] + 1e-8),
        "tp": tp,
        "n_active": n_active,
        "n_term_genes": n_term,
    }


def annotate_node_features(
    sae_model,
    npz_file,
    term_masks,
    gene_list=None,
    device="cpu",
    thresholds=(0.15, 0.3, 0.5),
    top_k=5,
    min_activation_frac=0.01,
    min_active_nodes=3,
    min_f1=0.05,
    group_cols=("disease", "cell_type"),
    batch_size=8192,
    verbose=True,
):
    """
    Annotate node-level SAE features with gene sets, one graph at a time.

    Steps 1–4 of the pipeline. For every graph: the rows belonging to it are run
    through the SAE, the activations are binarised at each relative threshold
    (``activation / global maximum > threshold``), every (feature, term) pair is
    scored by F1, the best threshold per pair is kept, and each feature keeps its
    ``top_k`` terms. The result is the ranked list the annotation is read off.

    Working per graph rather than over the whole file is both cheaper (a
    ``[n_genes, n_features]`` matrix instead of ``[n_rows, n_features]``) and the
    point of the analysis: a feature's F1 for a term is a per-sample measurement,
    whose mean and spread across graphs are what ``consensus_feature_terms`` and
    ``group_feature_terms`` then integrate.

    Parameters
    ----------
    sae_model : ShallowSAE
        Trained **node-level** SAE, matching the embedding dim of the file.
    npz_file : str or pathlib.Path
        Node-level export (``*_node.npz``).
    term_masks : dict or pd.DataFrame
        Output of ``geneSetAnnotation`` (with ``gene_list``) or of
        ``gene_set_matrix``.
    gene_list : list of str, optional
        Gene order of the masks; not needed when ``term_masks`` is a table.
    device : torch.device or str
    thresholds : iterable of float
        Relative activation thresholds to try, as in
        ``gea.analysis.gea_annotation``. A low threshold favours recall, a high
        one precision; keeping the best per pair lets each feature be read at the
        sharpness that actually separates the term.
    top_k : int
        Terms kept per feature per graph.
    min_activation_frac : float
        Features firing in fewer than this fraction of **all** rows of the file are
        dropped up front — a dead feature would otherwise be scored in every graph
        and dominate the multiple-testing burden.
    min_active_nodes : int
        Within a graph, a feature must fire on at least this many nodes to be
        scored. One firing node makes F1 look meaningful when it is noise.
    min_f1 : float
        Records below this F1 are discarded.
    group_cols : iterable of str
        Identity columns copied onto every record (present in the export as
        ``disease`` / ``cell_type``), so the result can be integrated per group.
    batch_size : int
        Rows per forward pass when computing the global feature maxima.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        One row per (graph, feature, term):
        ``graph_id``, ``*group_cols``, ``feature`` (int index into the SAE latent),
        ``term``, ``rank`` (1 = best term of that feature in that graph), ``f1``,
        ``f1_ceiling``, ``f1_of_ceiling``, ``precision``, ``recall``,
        ``fold_enrichment``, ``threshold``, ``n_active`` (nodes firing) and
        ``n_term_genes``.

    Notes
    -----
    ``f1`` means exactly what it says. It is ``2·tp/(2·tp + fp + fn)`` — there is no
    true-negative term, so the size of the gene universe does not enter it and does
    not depress it: a feature firing on precisely the 25 genes of a term, out of any
    number of genes, scores 1.0. A low F1 is therefore a real statement — the genes
    the feature fires on and the genes of the term only partly coincide — and the
    other columns say *how* they fail to coincide rather than excusing it:

    - ``precision`` vs ``recall`` — which side the loss is on. Low precision means
      the feature fires on genes outside the term (``fp``); low recall means the
      term's genes do not all fire it (``fn``). These call for opposite fixes.
    - ``f1_ceiling`` — the highest F1 reachable given only *how broadly the feature
      fires*: ``2·min(n_active, n_term)/(n_active + n_term)``. A feature firing on
      300 nodes against a 25-gene term has at least 275 false positives, so it
      cannot beat 0.15 however well-placed it is. This is a statement about the
      SAE's sparsity, and it is still purely about ``fp``/``fn``.
    - ``f1_of_ceiling`` — ``f1 / f1_ceiling``: how much of what its breadth allowed
      the feature actually got. Judge a feature by this rather than by ``f1`` alone.
    - ``fold_enrichment`` — precision over the term's prevalence. This one *does*
      depend on the size of the pool, and it answers a different question: is the
      feature better than firing at random? A feature can be 50× enriched and still
      overlap only a third of the term, so it never licenses a low F1.

    ``sae_health`` and ``term_neighbourhood_enrichment`` are the companion checks:
    together they separate a badly trained SAE from a space that never held the
    pathway signal from features that simply pick out a different gene set.
    """
    from gea.dataloader import EmbeddingDataset, load_embedding_metadata

    terms = _as_term_frame(term_masks, gene_list)
    meta = load_embedding_metadata(npz_file)
    if "gene" not in meta.columns:
        raise ValueError(
            f"{npz_file} has no 'gene' column — annotate_node_features needs a "
            "node-level export (export_embeddings level='node')."
        )

    embeddings = EmbeddingDataset(npz_file).embeddings
    maxima, frequency = feature_maxima(sae_model, embeddings, device, batch_size)

    alive = torch.nonzero(frequency >= min_activation_frac).reshape(-1)
    if not len(alive):
        raise ValueError(
            f"no feature fires in >= {min_activation_frac:.1%} of rows — the SAE "
            "may have collapsed, or min_activation_frac is too strict."
        )
    alive_maxima = maxima[alive]
    feature_ids = alive.cpu().numpy()
    if verbose:
        print(f"Alive: {len(alive)}/{sae_model.latent_dim} node features "
              f"(firing in >= {min_activation_frac:.1%} of rows)")

    # Term membership per row of the export, reached by gene position rather than
    # assuming the export is ordered gene-by-gene within each graph.
    gene_pos = {gene: i for i, gene in enumerate(terms.index)}
    term_arr = torch.as_tensor(terms.to_numpy(), device=device)
    term_names = np.asarray(terms.columns)
    try:
        row_gene = np.array([gene_pos[g] for g in meta["gene"]])
    except KeyError as err:
        raise ValueError(
            f"gene {err.args[0]!r} of the export is absent from the term masks; "
            "the masks must cover the whole node universe."
        ) from None

    group_cols = [c for c in group_cols if c in meta.columns]
    thresholds = list(thresholds)
    top_k = min(top_k, len(term_names))

    sae_model.to(device)
    sae_model.eval()

    graph_rows = meta.groupby("graph_id", sort=False).indices
    records = []

    with torch.no_grad():
        for graph_id, rows in graph_rows.items():
            z, _ = sae_model(embeddings[rows].to(device))
            relative = z[:, alive] / alive_maxima                # [n_nodes, n_alive]
            graph_terms = term_arr[row_gene[rows]]               # [n_nodes, n_terms]

            best_f1 = None
            for threshold in thresholds:
                scored = feature_term_f1(relative > threshold, graph_terms)
                # Features firing on too few nodes are not scored at this threshold
                too_sparse = (scored["n_active"] < min_active_nodes)[:, None]
                f1 = scored["f1"].masked_fill(too_sparse, 0.0)

                if best_f1 is None:
                    best_f1 = f1
                    best = {k: scored[k] for k in ("precision", "recall")}
                    best["n_active"] = scored["n_active"][:, None].expand_as(f1).clone()
                    best["threshold"] = torch.full_like(f1, threshold)
                else:
                    better = f1 > best_f1
                    best_f1 = torch.where(better, f1, best_f1)
                    best["precision"] = torch.where(better, scored["precision"], best["precision"])
                    best["recall"] = torch.where(better, scored["recall"], best["recall"])
                    best["n_active"] = torch.where(
                        better, scored["n_active"][:, None].expand_as(f1), best["n_active"]
                    )
                    best["threshold"] = torch.where(
                        better, torch.full_like(f1, threshold), best["threshold"]
                    )

            top_f1, top_terms = torch.topk(best_f1, k=top_k, dim=1)
            keep = (top_f1 >= min_f1).cpu().numpy()
            if not keep.any():
                continue

            feat_pos, rank = np.nonzero(keep)                    # positions in the top-k table
            term_pos = top_terms.cpu().numpy()[feat_pos, rank]
            gather = (feat_pos, term_pos)

            n_term_genes = graph_terms.sum(dim=0).cpu().numpy()[term_pos]
            precision = best["precision"].cpu().numpy()[gather]
            n_active = best["n_active"].cpu().numpy()[gather]
            f1 = top_f1.cpu().numpy()[feat_pos, rank]
            # Best F1 reachable given only how *broadly* the feature fires: even a
            # perfect feature that fires on 300 nodes cannot score above
            # 2·25/(300+25) = 0.15 on a 25-gene term.
            f1_ceiling = 2 * np.minimum(n_active, n_term_genes) / (n_active + n_term_genes)

            frame = pd.DataFrame({
                "graph_id": graph_id,
                "feature": feature_ids[feat_pos],
                "term": term_names[term_pos],
                "rank": rank + 1,
                "f1": f1,
                "f1_ceiling": f1_ceiling,
                "f1_of_ceiling": f1 / (f1_ceiling + 1e-12),
                "precision": precision,
                "recall": best["recall"].cpu().numpy()[gather],
                # Precision over the term's prevalence among the nodes: how many
                # times more likely a gene the feature fires on is to be in the term
                # than a gene drawn at random. It answers a different question from
                # F1 — a feature can be strongly enriched for a term and still
                # overlap only half of it.
                "fold_enrichment": precision / (n_term_genes / len(rows) + 1e-12),
                "threshold": best["threshold"].cpu().numpy()[gather],
                "n_active": n_active.astype(int),
                "n_term_genes": n_term_genes.astype(int),
            })
            for col in group_cols:
                frame[col] = meta[col].values[rows[0]]
            records.append(frame)

    if not records:
        raise ValueError(
            f"no (feature, term) pair reached F1 >= {min_f1} in any graph; try a "
            "lower min_f1, more thresholds, or broader gene sets."
        )

    per_graph = pd.concat(records, ignore_index=True)
    per_graph = per_graph[
        ["graph_id", *group_cols, "feature", "term", "rank", "f1", "f1_ceiling",
         "f1_of_ceiling", "precision", "recall", "fold_enrichment", "threshold",
         "n_active", "n_term_genes"]
    ]
    if verbose:
        print(f"{len(per_graph):,} (graph, feature, term) records — "
              f"{per_graph['feature'].nunique()} features, "
              f"{per_graph['term'].nunique()} terms, "
              f"{per_graph['graph_id'].nunique()} graphs")
    return per_graph


# ── Integration across graphs ────────────────────────────────────────────────

def consensus_feature_terms(per_graph, min_support=0.0, min_mean_f1=0.0):
    """
    Integrate the per-graph annotations into one score per (feature, term).

    Step 5. Two things make an annotation trustworthy, and they are kept apart:

    - ``mean_f1`` — how well the feature matches the term, averaged over the
      graphs where the term made that feature's top-k. It is a *conditional*
      mean, so on its own it says nothing about how often that happened.
    - ``support`` — the fraction of graphs where it happened at all.

    ``score = mean_f1 × support`` combines them, and is what the assignment and
    the plots rank on: a term that matches perfectly in three of 196 profiles
    scores below one that matches decently everywhere.

    Parameters
    ----------
    per_graph : pd.DataFrame
        Output of ``annotate_node_features``.
    min_support : float
        Drop pairs seen in fewer than this fraction of graphs.
    min_mean_f1 : float
        Drop pairs below this mean F1.

    Returns
    -------
    pd.DataFrame
        ``feature``, ``term``, ``n_graphs``, ``support``, ``mean_f1``, ``sd_f1``,
        ``max_f1``, ``mean_f1_of_ceiling``, ``mean_precision``, ``mean_recall``,
        ``mean_fold_enrichment``,
        ``n_top1`` (times the term was that feature's best), ``median_threshold``,
        ``n_term_genes``, ``score``. Sorted by ``score`` descending.
    """
    n_graphs = per_graph["graph_id"].nunique()

    consensus = (
        per_graph.groupby(["feature", "term"])
        .agg(
            n_graphs=("graph_id", "nunique"),
            mean_f1=("f1", "mean"),
            sd_f1=("f1", "std"),
            max_f1=("f1", "max"),
            mean_f1_of_ceiling=("f1_of_ceiling", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_fold_enrichment=("fold_enrichment", "mean"),
            n_top1=("rank", lambda r: int((r == 1).sum())),
            median_threshold=("threshold", "median"),
            n_term_genes=("n_term_genes", "median"),
        )
        .reset_index()
    )

    consensus["sd_f1"] = consensus["sd_f1"].fillna(0.0)   # a single graph has no spread
    consensus["support"] = consensus["n_graphs"] / n_graphs
    consensus["score"] = consensus["mean_f1"] * consensus["support"]
    consensus = consensus[
        (consensus["support"] >= min_support) & (consensus["mean_f1"] >= min_mean_f1)
    ]

    return consensus.sort_values("score", ascending=False).reset_index(drop=True)


def assign_feature_terms(consensus, min_score=0.0, min_support=0.0):
    """
    One annotation per feature: its best-scoring term, plus how clearly it won.

    The SAE "dictionary" — read it as *feature 37 encodes insulin secretion, in
    83% of profiles, F1 0.41*. ``margin`` is the gap to the feature's runner-up
    term: a large margin means the feature is specific to one concept, a margin
    near zero means it sits between several overlapping gene sets and the single
    label is a simplification (inspect ``consensus`` for that feature instead).

    Parameters
    ----------
    consensus : pd.DataFrame
        Output of ``consensus_feature_terms``.
    min_score, min_support : float
        Features whose best term does not clear these are left unannotated
        (dropped from the result).

    Returns
    -------
    pd.DataFrame
        One row per annotated feature: ``feature``, ``term``, ``score``,
        ``mean_f1``, ``sd_f1``, ``support``, ``mean_precision``, ``mean_recall``,
        ``n_term_genes``, ``margin``, ``n_candidate_terms``. Sorted by ``score``.
    """
    ranked = consensus.sort_values("score", ascending=False)
    grouped = ranked.groupby("feature", sort=False)

    best = grouped.head(1).set_index("feature")
    second = ranked.groupby("feature")["score"].apply(
        lambda s: s.iloc[1] if len(s) > 1 else 0.0
    )

    best = best.assign(
        margin=best["score"] - second.reindex(best.index).fillna(0.0),
        n_candidate_terms=grouped.size().reindex(best.index).values,
    ).reset_index()

    best = best[(best["score"] >= min_score) & (best["support"] >= min_support)]
    keep = ["feature", "term", "score", "mean_f1", "sd_f1", "support",
            "mean_f1_of_ceiling", "mean_precision", "mean_recall",
            "mean_fold_enrichment", "n_term_genes", "margin", "n_candidate_terms"]
    keep = [c for c in keep if c in best.columns]
    return best[keep].sort_values("score", ascending=False).reset_index(drop=True)


def group_feature_terms(per_graph, by="cell_type", min_graphs=3, **kwargs):
    """
    Run the integration separately within each phenotype or cell type.

    Support is then relative to the graphs *of that group*, so the same feature can
    come out as one concept in beta cells and another in stellate cells, or hold a
    term in T2D profiles and lose it in the controls. Comparing the groups is where
    a condition-specific concept shows up.

    Parameters
    ----------
    per_graph : pd.DataFrame
        Output of ``annotate_node_features``; must contain the ``by`` column.
    by : str
        Grouping column, typically ``'disease'`` or ``'cell_type'``.
    min_graphs : int
        Groups with fewer graphs than this are skipped — support estimated on two
        profiles is not a support.
    **kwargs
        Passed to ``consensus_feature_terms`` (``min_support``, ``min_mean_f1``).

    Returns
    -------
    pd.DataFrame
        ``consensus_feature_terms`` columns plus the ``by`` column and
        ``n_group_graphs``.
    """
    if by not in per_graph.columns:
        raise ValueError(f"{by!r} is not a column of per_graph; have {list(per_graph.columns)}")

    frames = []
    for group, subset in per_graph.groupby(by):
        n = subset["graph_id"].nunique()
        if n < min_graphs:
            print(f"skipping {group!r}: {n} graph(s) < min_graphs={min_graphs}")
            continue
        consensus = consensus_feature_terms(subset, **kwargs)
        consensus.insert(0, by, group)
        consensus["n_group_graphs"] = n
        frames.append(consensus)

    if not frames:
        raise ValueError(f"no {by} group has at least {min_graphs} graphs")

    return pd.concat(frames, ignore_index=True)


# ── Diagnostics: is the signal there, and did the SAE keep it? ───────────────
#
# When the F1 scores come out low there are three quite different causes, and they
# need to be told apart before any of them is acted on:
#
#   (a) the SAE is badly trained — dead dictionary, poor reconstruction, or so
#       little sparsity that a feature fires on everything. ``sae_health``.
#   (b) the pathway signal is not in the node embeddings at all, so no
#       decomposition of them could recover it. ``term_neighbourhood_enrichment``
#       measures it in the GNN space and in the SAE space side by side, and the
#       UMAP views show the same thing by eye.
#   (c) neither: the signal is there and the SAE found it, but the gene set a
#       feature picks out is not the curated gene set. That is what a low F1 says,
#       and ``precision`` vs ``recall`` says on which side — firing outside the term
#       or missing its genes. Nothing about the size of the gene universe softens
#       this: F1 has no true-negative term.
#
# Everything here works on **gene-level** matrices from ``gene_level_matrices``:
# one row per gene rather than per (graph, gene). The node universe is shared by
# every graph, so a term is a property of the gene, and collapsing the ~200 k rows
# to 1,030 genes is both the right unit for "do the genes of a pathway sit
# together?" and what makes a UMAP and a k-NN sweep cheap.

def gene_level_matrices(sae_model, npz_file, device="cpu", batch_size=8192,
                        firing_threshold=None, maxima=None, verbose=True):
    """
    Collapse a node-level export to one row per gene, in both spaces.

    Parameters
    ----------
    sae_model : ShallowSAE
        Trained node-level SAE. Pass ``None`` to skip the activation matrix.
    npz_file : str or pathlib.Path
        Node-level export (``*_node.npz``).
    device : torch.device or str
    batch_size : int
    firing_threshold : float, optional
        When given, also return ``firing_rate``: per (gene, feature), the fraction
        of that gene's rows whose *relative* activation exceeds this threshold —
        thresholded the same way ``annotate_node_features`` does, per row, before
        any averaging. Use this, not ``activation``, whenever the question is "which
        genes does this feature fire on?": averaging a sparse activation over ~200
        graphs and then comparing it to a global maximum shrinks almost every
        feature below any useful threshold and silently yields empty gene sets.
    maxima : array-like, optional
        Per-feature maxima the threshold is relative to (``sae_health``'s
        ``maxima``, or ``feature_maxima``'s first return value). Computed here when
        omitted and ``firing_threshold`` is given.
    verbose : bool

    Returns
    -------
    dict
        - ``genes`` : list of gene IDs, the row order of every matrix.
        - ``embedding`` : ``[n_genes, D]`` mean GNN node embedding per gene.
        - ``activation`` : ``[n_genes, n_features]`` mean SAE activation per gene
          (absent when ``sae_model`` is None). Good for UMAP and neighbourhoods,
          where only the direction of the vector matters.
        - ``firing_rate`` : ``[n_genes, n_features]`` fraction of the gene's rows
          where the feature fired above ``firing_threshold`` (only when that is
          given).
        - ``n_rows`` : ``[n_genes]`` graphs contributing to each gene.
        - ``gene_variance_explained`` : fraction of the variance of ``z_node``
          that gene identity accounts for. The rest is sample-to-sample variation
          (which profile the node came from), so a low value means the averaging
          throws away most of what the embedding encodes and the gene-level view
          is only part of the picture.
    """
    from gea.dataloader import EmbeddingDataset, load_embedding_metadata

    meta = load_embedding_metadata(npz_file)
    if "gene" not in meta.columns:
        raise ValueError(f"{npz_file} is not a node-level export (no 'gene' column)")

    embeddings = EmbeddingDataset(npz_file).embeddings
    genes = list(dict.fromkeys(meta["gene"].tolist()))
    position = {gene: i for i, gene in enumerate(genes)}
    row_gene = np.array([position[g] for g in meta["gene"]])
    counts = np.bincount(row_gene, minlength=len(genes))[:, None].astype(np.float32)

    X = embeddings.numpy()
    gene_embedding = np.zeros((len(genes), X.shape[1]), dtype=np.float32)
    np.add.at(gene_embedding, row_gene, X)
    gene_embedding /= counts

    total = X.var(axis=0).sum()
    within = ((X - gene_embedding[row_gene]) ** 2).mean(axis=0).sum()
    explained = float(1 - within / total)

    out = {
        "genes": genes,
        "embedding": gene_embedding,
        "n_rows": counts.ravel().astype(int),
        "gene_variance_explained": explained,
    }

    if sae_model is not None:
        sae_model.to(device)
        sae_model.eval()

        if firing_threshold is not None:
            if maxima is None:
                maxima, _ = feature_maxima(sae_model, embeddings, device, batch_size)
            scale = torch.as_tensor(np.asarray(maxima), dtype=torch.float32,
                                    device=device).clamp(min=1e-12)
            fired = np.zeros((len(genes), sae_model.latent_dim), dtype=np.float32)

        activation = np.zeros((len(genes), sae_model.latent_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(embeddings), batch_size):
                z, _ = sae_model(embeddings[start:start + batch_size].to(device))
                rows = row_gene[start:start + batch_size]
                np.add.at(activation, rows, z.cpu().numpy())
                if firing_threshold is not None:
                    hit = (z / scale > firing_threshold).float().cpu().numpy()
                    np.add.at(fired, rows, hit)
        activation /= counts
        out["activation"] = activation
        if firing_threshold is not None:
            out["firing_rate"] = fired / counts

    if verbose:
        print(f"{len(genes)} genes × {len(meta) // len(genes)} graphs | "
              f"gene identity explains {explained:.1%} of z_node variance "
              f"(the rest is which profile the node came from)")
    return out


def sae_health(sae_model, npz_file, device="cpu", batch_size=8192,
               min_activation_frac=0.01, verbose=True):
    """
    Is the SAE itself trained well? Reconstruction, sparsity and dictionary use.

    The first thing to check when annotation F1 comes out low, because all three
    failure modes of an SAE show up here:

    - **reconstruction** (``r2``) far below ~0.9 → the features are not a faithful
      basis for the embeddings, so nothing they say about the embeddings is safe.
    - **dead dictionary** (``n_dead`` near ``latent_dim``) → the sparsity penalty
      killed most units; the survivors have to be polysemantic, which caps how
      cleanly any one of them can match a single gene set.
    - **too little sparsity** (``l0_mean`` a large share of ``latent_dim``) → a
      feature that fires on most nodes cannot be specific to a 10-gene pathway; its
      precision is bounded by the term's prevalence no matter what it encodes.

    Parameters
    ----------
    sae_model : ShallowSAE
    npz_file : str or pathlib.Path
        Any export the SAE was trained on.
    device : torch.device or str
    batch_size : int
    min_activation_frac : float
        The ``annotate_node_features`` alive cutoff, reported here so the two
        agree.
    verbose : bool

    Returns
    -------
    dict
        ``latent_dim``, ``n_rows``, ``r2``, ``l0_mean``, ``l0_per_row``,
        ``n_dead``, ``n_alive``, ``frequency`` ``[n_features]``, ``maxima``
        ``[n_features]``, ``mean_when_firing`` ``[n_features]``,
        ``min_activation_frac``.
    """
    from gea.dataloader import EmbeddingDataset

    embeddings = EmbeddingDataset(npz_file).embeddings
    sae_model.to(device)
    sae_model.eval()

    latent = sae_model.latent_dim
    maxima = torch.zeros(latent, device=device)
    active = torch.zeros(latent, device=device)
    total = torch.zeros(latent, device=device)
    l0_per_row = np.empty(len(embeddings), dtype=np.int32)
    mean_x = embeddings.mean(dim=0).to(device)
    sse = sst = 0.0

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            xb = embeddings[start:start + batch_size].to(device)
            z, recon = sae_model(xb)
            fires = z > 0
            maxima = torch.maximum(maxima, z.max(dim=0).values)
            active += fires.sum(dim=0)
            total += z.sum(dim=0)
            l0_per_row[start:start + len(xb)] = fires.sum(dim=1).cpu().numpy()
            sse += ((recon - xb) ** 2).sum().item()
            sst += ((xb - mean_x) ** 2).sum().item()

    frequency = (active / len(embeddings)).cpu().numpy()
    health = {
        "latent_dim": latent,
        "n_rows": len(embeddings),
        "r2": 1 - sse / sst,
        "l0_mean": float(l0_per_row.mean()),
        "l0_per_row": l0_per_row,
        "n_dead": int((frequency == 0).sum()),
        "n_alive": int((frequency >= min_activation_frac).sum()),
        "frequency": frequency,
        "maxima": maxima.cpu().numpy(),
        "mean_when_firing": (total / active.clamp(min=1)).cpu().numpy(),
        "min_activation_frac": min_activation_frac,
    }

    if verbose:
        print(f"reconstruction R²      : {health['r2']:.3f}")
        print(f"L0 per node            : {health['l0_mean']:.1f} of {latent} features "
              f"({health['l0_mean'] / latent:.1%} of the dictionary fires per node)")
        print(f"never fires            : {health['n_dead']}/{latent}")
        print(f"fires in >= {min_activation_frac:.1%} of rows : {health['n_alive']}/{latent}")
        print(f"max activation         : median {np.median(health['maxima']):.3f}, "
              f"max {health['maxima'].max():.1f}")
    return health


def plot_sae_health(health, figsize=(13, 3.6)):
    """
    The three numbers of ``sae_health`` as distributions.

    Left: how often each feature fires, with the alive cutoff marked — the shape of
    this tells a healthy sparse dictionary (a broad band of features firing on a few
    percent of rows) from a collapsed one (a spike at zero and a few features
    firing on everything). Middle: how many features fire per node, i.e. the actual
    sparsity. Right: peak activation per feature, on a log scale, which is what the
    relative thresholds in ``annotate_node_features`` are taken against.

    Parameters
    ----------
    health : dict
        Output of ``sae_health``.
    figsize : tuple

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt
    from gea.analysis import _SEQ_BLUE, _INK, _MUTED

    freq = health["frequency"]
    maxima = health["maxima"]
    fires = freq > 0
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    bins = np.logspace(-6, 0, 40)
    axes[0].hist(np.clip(freq[fires], 1e-6, 1), bins=bins, color=_SEQ_BLUE[4])
    axes[0].axvline(health["min_activation_frac"], color=_INK, lw=1)
    axes[0].text(health["min_activation_frac"] * 1.15, axes[0].get_ylim()[1] * 0.92,
                 f"alive cutoff\n{health['n_alive']} features", fontsize=7, color=_INK,
                 va="top")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("fraction of nodes where the feature fires")
    axes[0].set_title(f"{int(fires.sum())} of {health['latent_dim']} features ever fire",
                      fontsize=10, loc="left", color=_INK)

    axes[1].hist(health["l0_per_row"], bins=40, color=_SEQ_BLUE[4])
    axes[1].axvline(health["l0_mean"], color=_INK, lw=1)
    axes[1].set_xlabel("features firing per node (L0)")
    axes[1].set_title(f"mean L0 {health['l0_mean']:.0f} / {health['latent_dim']}"
                      f"  ·  R² {health['r2']:.3f}", fontsize=10, loc="left", color=_INK)

    positive = maxima[maxima > 1e-8]
    axes[2].hist(positive, bins=np.logspace(np.log10(max(positive.min(), 1e-8)),
                                            np.log10(positive.max()), 40),
                 color=_SEQ_BLUE[4])
    axes[2].set_xscale("log")
    axes[2].set_xlabel("peak activation per feature")
    axes[2].set_title("dynamic range of the dictionary", fontsize=10, loc="left",
                      color=_INK)

    for ax in axes:
        ax.set_ylabel("features" if ax is not axes[1] else "nodes", color=_MUTED)
        ax.tick_params(labelsize=8, colors=_MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#eceae5")
    fig.tight_layout()
    return fig, axes


def term_neighbourhood_enrichment(matrices, term_masks, gene_list=None, k=15,
                                  spaces=("embedding", "activation")):
    """
    Is a gene set localised in each space, and how good is a given F1 on it?

    Measured in the **full** space rather than in a projection, so it is evidence
    rather than illustration:

    - ``share_<space>`` — of the ``k`` nearest neighbours of a term's genes, the
      fraction that also belong to the term. This is the precision an ideal *local*
      classifier of that term reaches at this neighbourhood scale, so it is the
      reference for the ``mean_precision`` of a feature: a feature at 0.33 against a
      share of 0.30 is already doing as well as the geometry allows.
    - ``enrichment_<space>`` — that share over the term's prevalence, i.e. over
      chance. Above ~3× the pathway is a real neighbourhood in this space; near 1×
      the space does not encode it and no decomposition of it could.
    - ``chance_f1`` — the F1 a feature firing on ``n_genes`` *random* nodes would
      score, which equals the prevalence. It is a floor, not a ceiling: the size of
      the gene universe sets how well a random feature does, and never caps how well
      a real one can do (F1 has no true-negative term, so a feature that fires on
      exactly a term's genes scores 1.0 whatever the pool size). Use it to say a
      feature beats random, never to explain away a low F1.

    Comparing the two spaces answers the SAE question directly: if enrichment in
    ``activation`` is at least as high as in ``embedding``, the SAE kept (or
    sharpened) the pathway structure, and low F1 values are a property of the
    labels, not of the SAE.

    Parameters
    ----------
    matrices : dict
        Output of ``gene_level_matrices``.
    term_masks : dict or pd.DataFrame
        ``geneSetAnnotation`` output (with ``gene_list``) or ``gene_set_matrix``.
    gene_list : list of str, optional
    k : int
        Neighbours per gene, cosine distance.
    spaces : iterable of str
        Which matrices of ``matrices`` to measure.

    Returns
    -------
    pd.DataFrame
        One row per term: ``term``, ``n_genes``, ``chance``/``chance_f1``
        (prevalence), and ``share_<space>`` + ``enrichment_<space>`` per space.
        Sorted by the enrichment of the first space.

    Notes
    -----
    No attempt is made to report a hard upper bound on F1. A tempting one — the
    best ball around a term's centroid — is not a bound at all: gene embeddings sit
    in a narrow cone, so ordering genes by similarity to a class centroid ranks by
    overall magnitude rather than by membership, and it scores *below* what real
    features achieve. ``share_<space>`` is the reference that behaves.
    """
    terms = _as_term_frame(term_masks, gene_list)
    terms = terms.reindex(matrices["genes"])
    if terms.isna().any().any():
        missing = terms.index[terms.isna().any(axis=1)]
        raise ValueError(
            f"{len(missing)} genes of the export are absent from the term masks "
            f"(e.g. {list(missing[:3])})"
        )
    T = terms.to_numpy().astype(bool)

    spaces = [s for s in spaces if s in matrices]
    out = pd.DataFrame({
        "term": list(terms.columns),
        "n_genes": T.sum(axis=0),
        "chance": T.mean(axis=0),
    })
    # A feature firing on n_genes random nodes: F1 = 2·n·p/(n + n) = p
    out["chance_f1"] = out["chance"]

    for space in spaces:
        M = matrices[space]
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        similarity = M @ M.T
        np.fill_diagonal(similarity, -np.inf)
        neighbours = np.argpartition(-similarity, k, axis=1)[:, :k]

        shares = [
            T[neighbours[np.nonzero(T[:, j])[0]], j].mean()
            for j in range(T.shape[1])
        ]
        out[f"share_{space}"] = shares
        out[f"enrichment_{space}"] = np.asarray(shares) / out["chance"].values

    return out.sort_values(f"enrichment_{spaces[0]}", ascending=False).reset_index(drop=True)


def umap_layouts(matrices, spaces=("embedding", "activation"), n_neighbors=15,
                 min_dist=0.3, metric="cosine", seed=42):
    """
    2-D UMAP of the gene-level matrices, one layout per space.

    Cosine distance and gene-level rows on purpose: the SAE activation vector is
    sparse and its magnitude varies by orders of magnitude between features, so
    direction is the comparable part, and one point per gene is what makes the
    layout about genes rather than about which profile a row came from.

    Parameters
    ----------
    matrices : dict
        Output of ``gene_level_matrices``.
    spaces : iterable of str
    n_neighbors, min_dist, metric : UMAP parameters
    seed : int

    Returns
    -------
    dict
        ``{space: np.ndarray [n_genes, 2]}``.
    """
    import umap

    layouts = {}
    for space in spaces:
        if space not in matrices:
            continue
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            metric=metric, random_state=seed)
        layouts[space] = reducer.fit_transform(matrices[space])
        print(f"UMAP of {space}: {matrices[space].shape} → 2D")
    return layouts


_SPACE_LABELS = {
    "embedding": "GNN node embedding",
    "activation": "SAE feature activations",
}


def plot_annotation_umap(layouts, matrices, term_masks, gene_list=None, terms=None,
                         top_terms=6, enrichment=None, point_size=9, figsize=None,
                         title="Do the genes of a pathway sit together?"):
    """
    Small multiples of the gene UMAP, one panel per gene set, one row per space.

    The visual answer to "is the pathway signal there, and did the SAE keep it?".
    Each panel highlights the genes of one term against every other gene, so the
    identity of a panel is carried by its title, not by a hue — which keeps this
    readable at any number of terms and under any colour vision. Comparing rows
    compares the two spaces on the same terms.

    A UMAP is a projection and can create or hide neighbourhoods, so read it
    together with ``term_neighbourhood_enrichment``, which measures the same thing
    in the full space.

    Parameters
    ----------
    layouts : dict
        Output of ``umap_layouts``.
    matrices : dict
        Output of ``gene_level_matrices`` (for the gene order).
    term_masks : dict or pd.DataFrame
    gene_list : list of str, optional
    terms : list of str, optional
        Terms to draw. Defaults to the ``top_terms`` best by ``enrichment``, or the
        largest terms when no enrichment table is given.
    top_terms : int
        How many panels when ``terms`` is not given.
    enrichment : pd.DataFrame, optional
        ``term_neighbourhood_enrichment`` output; used to pick the panels and to
        print each term's enrichment in its panel title.
    point_size : float
    figsize : tuple, optional
    title : str

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt
    from gea.analysis import _SEQ_BLUE, _INK, _MUTED

    frame = _as_term_frame(term_masks, gene_list).reindex(matrices["genes"])

    if terms is None:
        if enrichment is not None:
            column = next(c for c in enrichment.columns if c.startswith("enrichment_"))
            terms = list(enrichment.nlargest(top_terms, column)["term"])
        else:
            terms = list(frame.sum().nlargest(top_terms).index)
    spaces = [s for s in ("embedding", "activation") if s in layouts]

    n_rows, n_cols = len(spaces), len(terms)
    figsize = figsize or (2.35 * n_cols + 1.2, 2.55 * n_rows + 0.7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    lookup = enrichment.set_index("term") if enrichment is not None else None

    for r, space in enumerate(spaces):
        xy = layouts[space]
        for c, term in enumerate(terms):
            ax = axes[r][c]
            member = frame[term].to_numpy().astype(bool)

            # Emphasis, not categorical: context in the de-emphasis grey, the term's
            # genes in one accent hue, drawn on top with a surface ring so members
            # that overlap stay countable.
            ax.scatter(xy[~member, 0], xy[~member, 1], s=point_size * 0.55,
                       c="#dcd9d3", linewidths=0, rasterized=True)
            ax.scatter(xy[member, 0], xy[member, 1], s=point_size * 2.6,
                       c=_SEQ_BLUE[5], edgecolor="#fcfcfb", linewidths=0.7, zorder=3)

            if r == 0:
                caption = _short(term, 30)
                if lookup is not None:
                    column = next(c2 for c2 in lookup.columns
                                  if c2.startswith("enrichment_"))
                    caption += f"\n{int(member.sum())} genes"
                    caption += f" · {lookup.loc[term, column]:.1f}× kNN"
                else:
                    caption += f"\n{int(member.sum())} genes"
                ax.set_title(caption, fontsize=8, color=_INK)
            if c == 0:
                ax.set_ylabel(_SPACE_LABELS.get(space, space), fontsize=9, color=_INK)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#eceae5")

    if title:
        fig.suptitle(title, x=0.01, y=1.0, ha="left", fontsize=12, color="#0b0b0b")
    fig.text(0.01, -0.01,
             "grey = all other genes  ·  blue = genes of the panel's term  ·  "
             "UMAP on cosine distance, one point per gene",
             fontsize=7.5, color=_MUTED, ha="left")
    fig.tight_layout()
    return fig, axes


def dictionary_redundancy(sae_model, assigned, matrices, health,
                          min_rate=0.5, min_features=3):
    """
    Do the features that share a gene set duplicate each other?

    When several features are annotated with the same term, the natural suspicion is
    *feature splitting*: one concept smeared over near-duplicate dictionary
    directions, which would cap each copy's recall and so its F1. This measures it
    two ways, because the two can disagree:

    - **direction** — cosine between the features' ``W_dec`` rows. These are the
      vectors the SAE writes embeddings in, and they are unit-norm, so cosine is the
      whole story. Near 1 means genuine duplicates.
    - **support** — Jaccard overlap of the *gene sets the features fire on*. Two
      features can point in unrelated directions and still light up the same genes,
      which is redundancy of a different, milder kind: they encode different aspects
      of the same genes rather than the same aspect twice.

    Note what this makes of a UMAP of the dictionary: if the pairwise cosines are
    all near zero, the decoder directions are mutually near-orthogonal, there are no
    neighbourhoods, and *any* 2-D projection of them is a featureless blob — not a
    tuning problem, a property of the data. Check ``pairs`` before reaching for one.
    The UMAP that does work on SAE output is ``umap_layouts`` on the per-gene
    activation vectors, where the points are genes.

    Parameters
    ----------
    sae_model : ShallowSAE
    assigned : pd.DataFrame
        ``assign_feature_terms`` output.
    matrices : dict
        ``gene_level_matrices`` output, built with a ``firing_threshold`` so that it
        carries ``firing_rate``. The mean ``activation`` matrix is the wrong input
        here: averaged over ~200 graphs it falls below any relative threshold for
        most features, so gene sets come out empty.
    health : dict
        ``sae_health`` output, for the alive cutoff and firing frequency.
    min_rate : float
        A gene counts as "fired on" by a feature when it fires in at least this
        fraction of the graphs, i.e. consistently rather than in one profile.
    min_features : int
        Only summarise terms carrying at least this many features.

    Returns
    -------
    dict
        - ``pairs`` : cosine of every pair of alive decoder directions.
        - ``same_term`` : the subset of those pairs that share an assigned term.
        - ``per_term`` : DataFrame with ``term``, ``n_features``,
          ``median_cosine``, ``max_cosine``, ``median_jaccard``, ``max_jaccard``,
          ``median_genes_fired`` — one row per term with ``min_features`` features.
    """
    directions = sae_model.W_dec.detach().cpu().numpy()
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)

    alive = np.nonzero(health["frequency"] >= health["min_activation_frac"])[0]
    cosine = directions[alive] @ directions[alive].T
    upper = np.triu_indices(len(alive), 1)
    position = {feature: i for i, feature in enumerate(alive)}

    if "firing_rate" not in matrices:
        raise ValueError(
            "matrices has no 'firing_rate': rebuild it with "
            "gene_level_matrices(..., firing_threshold=0.15). The mean activation "
            "matrix cannot stand in — averaged over every graph it drops below any "
            "relative threshold for most features and the gene sets come out empty."
        )
    firing_rate = matrices["firing_rate"]

    same_term, rows = [], []
    for term, group in assigned.groupby("term"):
        features = [f for f in group["feature"].astype(int) if f in position]
        if len(features) < 2:
            continue

        index = [position[f] for f in features]
        block = cosine[np.ix_(index, index)][np.triu_indices(len(index), 1)]
        same_term.extend(block.tolist())

        if len(features) < min_features:
            continue

        fired = [set(np.nonzero(firing_rate[:, f] >= min_rate)[0]) for f in features]
        jaccard = [
            len(a & b) / max(len(a | b), 1)
            for i, a in enumerate(fired) for b in fired[i + 1:]
        ]
        rows.append({
            "term": term,
            "n_features": len(features),
            "median_cosine": float(np.median(block)),
            "max_cosine": float(block.max()),
            "median_jaccard": float(np.median(jaccard)),
            "max_jaccard": float(max(jaccard)),
            "median_genes_fired": float(np.median([len(s) for s in fired])),
        })

    per_term = (
        pd.DataFrame(rows).sort_values("n_features", ascending=False).reset_index(drop=True)
        if rows else pd.DataFrame(columns=["term", "n_features", "median_cosine",
                                           "max_cosine", "median_jaccard",
                                           "max_jaccard", "median_genes_fired"])
    )
    return {
        "pairs": cosine[upper],
        "same_term": np.asarray(same_term),
        "per_term": per_term,
    }


def consensus_adjacency(graphs, n_genes=None, min_fraction=0.5):
    """
    The gene–gene network shared by most sample-specific graphs.

    LIONESS thresholds each profile's edges independently, so no single graph is
    "the" network. An edge kept in at least ``min_fraction`` of the profiles is the
    part of the PPI prior that survived co-expression filtering across the cohort —
    the right backbone to ask whether a set of genes is a module.

    Parameters
    ----------
    graphs : list of torch_geometric.data.Data
        The PyG graphs (``t2d_graphs.pt``), all over the same node universe.
    n_genes : int, optional
        Nodes per graph; taken from the first graph when omitted.
    min_fraction : float
        Fraction of graphs an edge must appear in.

    Returns
    -------
    np.ndarray
        ``[n_genes, n_genes]`` boolean, symmetric, no self-loops.
    """
    n_genes = n_genes or int(graphs[0].x.shape[0])
    count = np.zeros((n_genes, n_genes), dtype=np.int32)
    for graph in graphs:
        edges = graph.edge_index.cpu().numpy()
        count[edges[0], edges[1]] += 1

    adjacency = count >= min_fraction * len(graphs)
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, False)
    print(f"consensus network: {int(adjacency.sum() // 2):,} edges over {n_genes} genes "
          f"(density {adjacency.sum() / (n_genes * (n_genes - 1)):.4f}), "
          f"edges kept in >= {min_fraction:.0%} of {len(graphs)} graphs")
    return adjacency


def feature_gene_set(matrices, feature, min_rate=0.25):
    """
    The genes a feature consistently fires on: those firing in at least
    ``min_rate`` of the graphs, from ``gene_level_matrices(..., firing_threshold=…)``.

    Returns
    -------
    np.ndarray
        Row indices into ``matrices["genes"]``.
    """
    if "firing_rate" not in matrices:
        raise ValueError("rebuild matrices with gene_level_matrices(..., firing_threshold=0.15)")
    return np.nonzero(matrices["firing_rate"][:, int(feature)] >= min_rate)[0]


def feature_gene_breakdown(matrices, feature, term_masks, term, gene_list=None,
                           min_rate=0.25, symbols=None):
    """
    Which genes a feature gets right, wrong, and misses for one term — by name.

    The qualitative counterpart of the F1 columns, and usually the fastest way to
    see *what* a feature encodes: a run of same-family genes among the false
    positives normally means the feature has found a real program that the curated
    term draws a different boundary around.

    Parameters
    ----------
    matrices : dict
        ``gene_level_matrices`` output built with a ``firing_threshold``.
    feature : int
    term_masks : dict or pd.DataFrame
    term : str
    gene_list : list of str, optional
    min_rate : float
    symbols : dict, optional
        ``{gene id: symbol}`` for readable output.

    Returns
    -------
    dict
        ``tp``, ``fp``, ``fn`` — lists of gene names (symbols when given).
    """
    frame = _as_term_frame(term_masks, gene_list).reindex(matrices["genes"])
    genes = np.asarray(matrices["genes"])
    fired = set(feature_gene_set(matrices, feature, min_rate).tolist())
    members = set(np.nonzero(frame[term].to_numpy())[0].tolist())

    def names(idx):
        return [str(symbols.get(g, g) if symbols else g) for g in genes[sorted(idx)]]

    return {"tp": names(fired & members), "fp": names(fired - members),
            "fn": names(members - fired)}


def term_module_coherence(assigned, matrices, term_masks, adjacency, gene_list=None,
                          min_rate=0.25, min_tp=3, n_null=300, seed=0):
    """
    Is the part of a pathway a feature matches a **module** of the gene network?

    The test behind the observation that features seem to fire on *submodules* of
    pathways — a complex, a reaction, one arm of a process — rather than on whole
    pathways. If that is what is happening, the genes a feature gets right are not
    an arbitrary slice of the term: they are the slice that is wired together.

    For each annotated (feature, term) pair it takes the true positives — the term's
    genes the feature consistently fires on — and compares the edge density of the
    subgraph they induce against random subsets *of the same term* with the same
    size. Sampling the null from within the term is what makes this a test of
    submodule structure rather than of pathway coherence: a pathway that is dense
    overall would otherwise look significant automatically.

    Parameters
    ----------
    assigned : pd.DataFrame
        ``assign_feature_terms`` output (or any frame with ``feature`` and ``term``).
    matrices : dict
        ``gene_level_matrices`` output built with a ``firing_threshold``.
    term_masks : dict or pd.DataFrame
    adjacency : np.ndarray
        ``[n_genes, n_genes]`` boolean, from ``consensus_adjacency``, in the row
        order of ``matrices["genes"]``.
    gene_list : list of str, optional
    min_rate : float
        A gene counts as fired on when it fires in this fraction of graphs. With
        several cell types in the cohort, a module restricted to one of them fires
        in only a fraction of the profiles, so 0.5 is often too strict.
    min_tp : int
        Pairs with fewer true positives than this are skipped — three genes is the
        smallest set whose density means anything.
    n_null : int
        Random subsets drawn per pair.
    seed : int

    Returns
    -------
    pd.DataFrame
        Per pair: ``feature``, ``term``, ``n_term``, ``n_fired``, ``tp``,
        ``tp_density`` (edge density among the matched genes), ``null_density``
        (mean over random same-size subsets of the term), ``whole_term_density``,
        ``enrichment`` (``tp_density / null_density``) and ``p_emp`` (fraction of
        random subsets at least as dense). Sorted by ``p_emp``.
    """
    frame = _as_term_frame(term_masks, gene_list).reindex(matrices["genes"])
    rng = np.random.default_rng(seed)

    def density(index):
        index = np.asarray(sorted(index))
        if len(index) < 2:
            return np.nan
        block = adjacency[np.ix_(index, index)]
        return block.sum() / (len(index) * (len(index) - 1))

    rows = []
    for _, row in assigned.iterrows():
        feature, term = int(row["feature"]), row["term"]
        members = np.nonzero(frame[term].to_numpy())[0]
        fired = set(feature_gene_set(matrices, feature, min_rate).tolist())
        tp = sorted(fired & set(members.tolist()))
        if len(tp) < min_tp:
            continue

        observed = density(tp)
        null = np.array([
            density(rng.choice(members, len(tp), replace=False)) for _ in range(n_null)
        ], dtype=float)
        mean_null = float(np.nanmean(null))

        rows.append({
            "feature": feature, "term": term, "n_term": len(members),
            "n_fired": len(fired), "tp": len(tp), "tp_density": observed,
            "null_density": mean_null, "whole_term_density": density(members),
            "enrichment": observed / mean_null if mean_null else np.nan,
            "p_emp": float(np.mean(null >= observed)),
        })

    if not rows:
        raise ValueError(
            f"no annotated pair has >= {min_tp} genes firing in >= {min_rate:.0%} of "
            "graphs; lower min_rate, or check that matrices carries firing_rate."
        )
    return pd.DataFrame(rows).sort_values("p_emp").reset_index(drop=True)


def plot_term_module_coherence(coherence, top_n=14, figsize=None,
                               title="Do features match a wired-together part of the pathway?"):
    """
    ``term_module_coherence`` as a dumbbell chart: for each (feature, term) pair,
    the density of a random same-size subset of the term against the density of the
    genes the feature actually matched.

    A long rightward jump means the matched genes are far more interconnected than an
    arbitrary slice of the same pathway — i.e. the feature found a submodule. Pairs
    whose dots sit on top of each other match the pathway diffusely.

    Parameters
    ----------
    coherence : pd.DataFrame
        Output of ``term_module_coherence``.
    top_n : int
        Pairs to show, most significant first.
    figsize : tuple, optional
    title : str

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from gea.analysis import _SEQ_BLUE, _INK, _MUTED

    data = coherence.head(top_n).iloc[::-1]
    labels = [f"f{int(f)} · {_short(t, 30)}  ({int(tp)}/{int(n)})"
              for f, t, tp, n in zip(data["feature"], data["term"], data["tp"], data["n_term"])]
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=figsize or (9.5, max(3.0, 0.42 * len(data))))
    ax.hlines(y, data["null_density"], data["tp_density"], color="#dcd9d3", lw=2.4,
              zorder=1)
    ax.scatter(data["null_density"], y, s=52, color=_SEQ_BLUE[2], zorder=2,
               edgecolor="#fcfcfb", linewidths=0.8,
               label="random same-size subset of the term")
    ax.scatter(data["tp_density"], y, s=52, color=_SEQ_BLUE[6], zorder=3,
               edgecolor="#fcfcfb", linewidths=0.8,
               label="the genes the feature matched")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, color=_INK)
    ax.set_xlabel("edge density of the induced subgraph (consensus network)", color=_INK)
    ax.set_xlim(left=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8, colors=_MUTED)
    ax.grid(axis="x", color="#eceae5", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#eceae5")
    # Above the plot rather than inside it: every row's dumbbell reaches to the
    # right, so any in-axes corner eventually collides with a mark.
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK, ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.4)
    ax.set_title(title, loc="left", fontsize=11.5, color="#0b0b0b", pad=26)
    fig.tight_layout()
    return fig, ax


def plot_dictionary_redundancy(redundancy, top_terms=10, figsize=(12.5, 4.2)):
    """
    ``dictionary_redundancy`` as two panels.

    Left: how similar the dictionary directions are — every alive pair against just
    the pairs that share a gene set. If the two distributions sit on top of each
    other near zero, features sharing a term are no more alike than any two
    features, and low F1 is not duplication. Two distributions, one hue in two
    shades, because they measure the same quantity on two subsets.

    Right: support overlap per term — the median Jaccard of the gene sets the
    features fire on, for the terms carrying the most features. This is the
    redundancy that survives when directions are orthogonal.

    Parameters
    ----------
    redundancy : dict
        Output of ``dictionary_redundancy``.
    top_terms : int
    figsize : tuple

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt
    from gea.analysis import _SEQ_BLUE, _INK, _MUTED

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                            gridspec_kw={"width_ratios": [1.0, 1.25]})

    bins = np.linspace(-1, 1, 81)
    axes[0].hist(redundancy["pairs"], bins=bins, density=True, color=_SEQ_BLUE[2],
                 label=f"all alive pairs (n={len(redundancy['pairs']):,})")
    if len(redundancy["same_term"]):
        axes[0].hist(redundancy["same_term"], bins=bins, density=True,
                     histtype="step", lw=1.6, color=_SEQ_BLUE[6],
                     label=f"pairs sharing a term (n={len(redundancy['same_term']):,})")
    axes[0].set_xlabel("cosine between decoder directions")
    axes[0].set_ylabel("density", color=_MUTED)
    axes[0].set_title(
        f"median {np.median(redundancy['pairs']):+.3f}  ·  "
        f"{int((redundancy['pairs'] > 0.7).sum())} of {len(redundancy['pairs']):,} pairs above 0.7",
        fontsize=10, loc="left", color=_INK)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=_INK)

    per_term = redundancy["per_term"].head(top_terms).iloc[::-1]
    if len(per_term):
        labels = [f"{_short(t, 34)}  ({n})"
                  for t, n in zip(per_term["term"], per_term["n_features"])]
        axes[1].barh(labels, per_term["median_jaccard"], color=_SEQ_BLUE[4], height=0.7)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("median Jaccard of the genes the features fire on")
        axes[1].set_title("support overlap between features sharing a term "
                          "(n features)", fontsize=10, loc="left", color=_INK)
        axes[1].tick_params(axis="y", labelsize=7.5, colors=_INK, length=0)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8, colors=_MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#eceae5")
    fig.tight_layout()
    return fig, axes


# ── Visualisation ────────────────────────────────────────────────────────────

def _short(term, width=48):
    """Library-prefixed term names are long; keep the informative tail."""
    term = str(term).split(" | ")[-1]
    return term if len(term) <= width else term[: width - 1] + "…"


def plot_feature_term_heatmap(
    consensus,
    value="score",
    n_features=30,
    n_terms=25,
    figsize=None,
    title="Node-level SAE features × gene sets",
):
    """
    Clustered heatmap of the feature–term annotation matrix.

    Rows and columns are both clustered, so features that share a concept — the
    SAE having split one biological process across several units — land next to
    each other, as do terms that no feature can tell apart.

    Parameters
    ----------
    consensus : pd.DataFrame
        Output of ``consensus_feature_terms``.
    value : str
        Cell value: ``'score'`` (F1 × support, default), ``'mean_f1'``,
        ``'support'``, ``'mean_precision'``, ``'mean_recall'``.
    n_features, n_terms : int
        Keep the top features and terms by best ``value``. Pairs never scored are
        drawn as 0.
    figsize : tuple, optional
    title : str

    Returns
    -------
    g : seaborn.matrix.ClusterGrid
    matrix : pd.DataFrame
        The features × terms table actually drawn.
    """
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from gea.analysis import _SEQ_BLUE, _INK, _MUTED

    top_features = (
        consensus.groupby("feature")[value].max().nlargest(n_features).index
    )
    subset = consensus[consensus["feature"].isin(top_features)]
    top_terms = subset.groupby("term")[value].max().nlargest(n_terms).index
    subset = subset[subset["term"].isin(top_terms)]

    matrix = (
        subset.pivot_table(index="feature", columns="term", values=value, aggfunc="max")
        .fillna(0.0)
    )
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(
            f"only {matrix.shape[0]} feature(s) × {matrix.shape[1]} term(s) to draw; "
            "loosen the filters in consensus_feature_terms."
        )

    figsize = figsize or (min(18.0, 6.0 + 0.34 * matrix.shape[1]),
                          min(16.0, 3.5 + 0.26 * matrix.shape[0]))
    g = sns.clustermap(
        matrix,
        cmap=LinearSegmentedColormap.from_list("gea_sequential", _SEQ_BLUE),
        figsize=figsize,
        xticklabels=[_short(t) for t in matrix.columns],
        yticklabels=[f"feature_{f}" for f in matrix.index],
        dendrogram_ratio=(0.12, 0.10),
        cbar_pos=(0.02, 0.70, 0.025, 0.13),
        linewidths=0.4,
        linecolor="#fcfcfb",
    )
    g.ax_cbar.set_title(value.replace("_", " "), fontsize=8, color=_INK, loc="left", pad=6)
    g.ax_cbar.tick_params(labelsize=7, colors=_MUTED, length=2)
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="x", labelsize=7, colors=_MUTED, rotation=90)
    g.ax_heatmap.tick_params(axis="y", labelsize=7, colors=_INK)
    if title:
        # Above the figure rather than in it: the column dendrogram reaches the top
        g.figure.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=13, color="#0b0b0b")

    return g, matrix


def plot_feature_annotation_bars(assigned, top_n=20, figsize=None,
                                 title="What each node-level SAE feature encodes"):
    """
    The assigned dictionary as a bar chart: one bar per feature, labelled with its
    term, length = mean F1, and the support printed at the end of the bar.

    Colour carries nothing here — a single hue for magnitude — because the
    identity of each bar is already written next to it.

    Parameters
    ----------
    assigned : pd.DataFrame
        Output of ``assign_feature_terms``.
    top_n : int
        Features to show, by score.
    figsize : tuple, optional
    title : str

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from gea.analysis import _SEQ_BLUE, _INK, _MUTED

    top = assigned.nlargest(top_n, "score").iloc[::-1]
    labels = [f"feature_{f} · {_short(t, 44)}" for f, t in zip(top["feature"], top["term"])]

    fig, ax = plt.subplots(figsize=figsize or (10, max(3.0, 0.42 * len(top))))
    ax.barh(labels, top["mean_f1"], color=_SEQ_BLUE[4], height=0.72,
            xerr=top["sd_f1"], error_kw=dict(ecolor=_MUTED, lw=0.8, capsize=2))

    # Support sits past the error bar, in one column, so the numbers read as a
    # column of their own rather than colliding with the whiskers
    span = float((top["mean_f1"] + top["sd_f1"]).max()) or 1.0
    for y, support in enumerate(top["support"]):
        ax.text(span * 1.04, y, f"{support:.0%} of graphs",
                va="center", fontsize=7, color=_MUTED)

    ax.set_xlim(0, span * 1.30)
    ax.set_xlabel("mean F1(term | feature) across graphs", color=_INK)
    ax.tick_params(axis="y", labelsize=8, colors=_INK, length=0)
    ax.tick_params(axis="x", labelsize=8, colors=_MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b", pad=10)
    fig.tight_layout()
    return fig, ax


def plot_group_term_dotplot(grouped, by="cell_type", features=None, top_n_terms=15,
                            value="mean_f1", figsize=None, title=None):
    """
    Dot plot of the group-stratified annotations: terms × groups, dot size =
    support, colour = F1.

    This is the view that answers "is this concept everywhere or only in one
    condition?" — a term with big dark dots in one column and nothing in the
    others is a group-specific concept, while a row of even dots is a
    housekeeping one.

    Parameters
    ----------
    grouped : pd.DataFrame
        Output of ``group_feature_terms``.
    by : str
        The grouping column of ``grouped`` (its x axis).
    features : iterable of int, optional
        Restrict to these SAE features. By default every feature contributes and
        each term is summarised by its best feature in that group.
    top_n_terms : int
        Terms to show, by best ``value`` across groups.
    value : str
        Colour scale: ``'mean_f1'`` (default) or ``'score'``.
    figsize : tuple, optional
    title : str, optional

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from gea.analysis import _SEQ_BLUE, _INK, _MUTED

    data = grouped if features is None else grouped[grouped["feature"].isin(list(features))]
    if data.empty:
        raise ValueError("no rows left to plot; check `features`")

    best = (
        data.sort_values(value, ascending=False)
        .groupby([by, "term"], as_index=False)
        .first()
    )
    top_terms = best.groupby("term")[value].max().nlargest(top_n_terms).index
    best = best[best["term"].isin(top_terms)]

    groups = sorted(best[by].unique())
    terms = list(best.groupby("term")[value].max().sort_values().index)
    x = best[by].map({g: i for i, g in enumerate(groups)})
    y = best["term"].map({t: i for i, t in enumerate(terms)})

    fig, ax = plt.subplots(
        figsize=figsize or (max(6.0, 1.6 * len(groups) + 4.0), max(3.5, 0.34 * len(terms)))
    )
    dots = ax.scatter(
        x, y, s=30 + 320 * best["support"], c=best[value],
        cmap=LinearSegmentedColormap.from_list("gea_sequential", _SEQ_BLUE),
        edgecolor=_MUTED, linewidth=0.4,
    )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_short(g, 22) for g in groups], rotation=30, ha="right")
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([_short(t) for t in terms])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(-0.8, len(terms) - 0.2)
    ax.grid(axis="y", color="#eceae5", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=_INK, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#eceae5")

    # Size and colour mean different things, so each needs its own key. The
    # colourbar is pushed right to leave the gap the size legend sits in.
    bar = fig.colorbar(dots, ax=ax, shrink=0.45, pad=0.30, anchor=(0.0, 1.0))
    bar.set_label(value.replace("_", " "), fontsize=8, color=_INK)
    bar.ax.tick_params(labelsize=7, colors=_MUTED, length=2)

    for support in (0.25, 0.5, 1.0):
        ax.scatter([], [], s=30 + 320 * support, c=_SEQ_BLUE[4],
                   edgecolor=_MUTED, linewidth=0.4, label=f"{support:.0%}")
    ax.legend(title="support", loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8, title_fontsize=8, labelspacing=1.3,
              handletextpad=1.0, borderpad=0.0)

    ax.set_title(title or f"Gene-set annotations per {by.replace('_', ' ')}",
                 loc="left", fontsize=12, color="#0b0b0b", pad=10)
    fig.tight_layout()
    return fig, ax
