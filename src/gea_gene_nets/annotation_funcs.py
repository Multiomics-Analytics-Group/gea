import torch
import numpy as np


# INPUT: 
#   - SAEs features at the graph-, node-, and edge-levels
#   - EmbeddingDataset object, with the embeddings identity and relevant metadata (e.g. what gene is it? what kind of relation the edge is encoding?)
# 
# THE NODE UNIVERSE:
# 
# Ok so the logic here is that we have our SAEs activation features ([n_graphs * n_nodes, n_features]) amd metadata like labels, node_id, graph_id, etc.
# We want to compute an enrichment of each SAEs feature per graph, and then we can think on how to integrate them.
# Then to be able to do that we have to follow these steps:
#
#   1. For each graph, we need to get the SAEs features and the corresponding metadata (e.g. node_id, graph_id, etc.)
#   2. For the universe of nodes (genes), we need to fetch the enrichment terms (e.g. GO terms, KEGG pathways, etc.)
#   3. For one graph: Per term, compute the F1-score for all the SAEs features activation profile (F1-score(term | feature))
#   4. Create a ranked list of F1-scores per feature, and assign each of the alive features a term (with confident score)
#   5. Think on how to integrate the results across graphs, and how to visualize them (e.g. heatmaps, barplots, etc.)
# 