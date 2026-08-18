# The __init__.py file is loaded when the package is loaded.
# It is used to indicate that the directory in which it resides is a Python package
from importlib import metadata

__version__ = metadata.version("gea")

from gea.utils import compress_embeddings_pca
from gea.gea import set_seed, seeded_generator, seed_worker

from gea.analysis import (
    # Activation extraction
    extract_graph_activations,
    extract_node_activations,
    extract_edge_activations,
    extract_sae_activations,
    # DFA
    filter_dead_features,
    differential_feature_activation,
    volcano_plot,
    plot_feature_activation_heatmap,
    plot_sae_feature_clustermap,
    feature_coactivation,
    # Explainability pipeline
    attribute_nodes_to_graph_feature,
    get_top_node_concepts,
    get_top_edge_concepts,
    explain_graph_feature,
    # Lower-level subgraph utilities
    trace_feature_to_subgraph,
    plot_feature_subgraph,
    # Gene set extraction & ORA
    get_attribution_gene_set,
    get_concept_gene_set,
    run_enrichment,
    label_features_by_genes,
)
