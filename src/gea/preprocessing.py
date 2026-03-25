# gea/preprocessing.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from gea.utils import get_ppi_edges
import torch
from torch_geometric.data import Data


def filter_cpm(
    counts_df: pd.DataFrame, cpm_threshold=1.0, min_frac=0.1
) -> pd.DataFrame:
    """
    Function used to filter out lowly expressed genes based on counts per million (CPM).

    Parameters
    ----------
    counts_df: pd.DataFrame
        A DataFrame containing raw count data with genes as rows and samples as columns.
    cpm_threshold: float
        The CPM threshold below which genes will be filtered.
    min_frac: float
        The minimum fraction of samples that must have CPM above the threshold for a gene to be retained.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only genes that meet the CPM criteria.
    """
    # Calculate total counts for each gene
    lib_size = counts_df.sum(axis=0)
    # Calculate CPM
    cpm = counts_df.div(lib_size, axis=1) * 1e6
    # Filter genes below threshold
    keep = (cpm >= cpm_threshold).sum(axis=1) >= np.ceil(min_frac * counts_df.shape[1])

    return counts_df.loc[keep, :]


def filter_var(counts_df: pd.DataFrame, pct=0.5) -> pd.DataFrame:
    """
    Function used to filter out lowly variable genes based on variance.

    Parameters
    ----------
    counts_df: pd.DataFrame
        A DataFrame containing raw count data with genes as rows and samples as columns.
    pct: float
        The percentage of the maximum variance to use as the threshold.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only genes that meet the variance criteria.
    """
    # Calculate total counts for each gene
    lib_size = counts_df.sum(axis=0)
    # Calculate CPM and log-transform
    cpm = counts_df.div(lib_size, axis=1) * 1e6
    logcpm = np.log2(cpm + 1.0)
    # Calculate variance for each gene across samples
    gene_var = logcpm.var(axis=1)
    # logcpm percentile - keep top pct counts
    q = 1.0 - pct
    cutoff = gene_var.quantile(q)
    keep = gene_var >= cutoff

    return counts_df.loc[keep, :]


def filter_genes(
    counts_df: pd.DataFrame, cpm_threshold=1.0, min_frac=0.1, pct=0.5
) -> pd.DataFrame:
    """
    Function used to filter out lowly expressed and lowly variable genes.

    Parameters
    ----------
    counts_df: pd.DataFrame
        A DataFrame containing raw count data with genes as rows and samples as columns.
    cpm_threshold: float
        The CPM threshold below which genes will be filtered.
    min_frac: float
        The minimum fraction of samples that must have CPM above the threshold for a gene to be retained.
    pct: float
        The percentage of the maximum variance to use as the threshold for filtering lowly variable genes.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only genes that meet both the CPM and variance criteria.
    """
    filtered_cpm = filter_cpm(counts_df, cpm_threshold, min_frac)
    filtered_cpm_var = filter_var(filtered_cpm, pct)

    return filtered_cpm_var


def filter_ppi_nodes(counts_df: pd.DataFrame, ppi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function used to filter the count matrix to include only genes present in the PPI network.

    Parameters
    ----------
    counts_df: pd.DataFrame
        A DataFrame containing count data (preferably filtered) with genes as rows and samples as columns.
    ppi_df: pd.DataFrame
        A DataFrame containing the PPI network with columns for the interacting proteins.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only genes where proteins are present in the PPI network.
    """
    # Get unique gene symbols from the PPI network
    ppi_gene_symbols = set(ppi_df["preferredName_A"]).union(ppi_df["preferredName_B"])
    # Filter the count matrix to include only genes present in the PPI network
    gene_data_in_ppi = counts_df[counts_df.index.isin(ppi_gene_symbols)]

    return gene_data_in_ppi


def normalize_counts(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function used to normalize count data using log2(CPM + 1) transformation.

    Parameters
    ----------
    counts_df: pd.DataFrame
        A DataFrame containing count data with genes as rows and samples as columns.

    Returns
    -------
    pd.DataFrame
        A normalized DataFrame where counts have been transformed to log2(CPM + 1). Rows and columns are switched.
    """
    # Gene count data into transponse format for normalization
    counts_df_t = counts_df.copy()
    counts_df_t.index.names = ["samples"]
    counts_df_t = counts_df_t.T

    # Normalize data (log2 CPM)
    lib_size = counts_df_t.sum(axis=1)
    if (lib_size == 0).any():
        raise ValueError("One or more samples have zero library size (not possible)")
    cpm = counts_df_t.div(lib_size, axis=0) * 1e6
    norm_data = np.log2(cpm + 1.0)

    return norm_data


def merge_metadata(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    counts_on="samples",
    meta_on="BioSample",
    meta_cols="source_name",
) -> pd.DataFrame:
    """
    Function used to merge the count matrix with sample metadata.

    Parameters
    ----------
    counts_df: pd.DataFrame
        A DataFrame containing count data with samples as rows and genes as columns.
    metadata_df: pd.DataFrame
        A DataFrame containing sample metadata with samples as rows and metadata variables as columns.
    counts_on: str
        The column name to use as the key for merging the count data (default is "samples").
    meta_on: str
        The column name to use as the key for merging the metadata (default is "BioSample").
    meta_cols: str or list
        The column name(s) from the metadata to include in the merged DataFrame (default is "source_name").

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing both count data and sample metadata.
    """
    rel_metadata = metadata_df[[meta_on, meta_cols]].rename(
        columns={meta_on: counts_on}
    )
    joint_data = rel_metadata.merge(
        counts_df, left_on=counts_on, right_index=True, how="inner"
    )

    return joint_data.set_index(counts_on)


def get_corr_matrix(counts_df: pd.DataFrame, group_by: str):
    """
    Function used to calculate the gene-gene correlation matrix from the count data (Pearson correlation).

    Parameters
    ----------
    counts_df: pd.DataFrame
        A DataFrame containing count data with genes as rows and samples as columns.
    group_by: str
        The column name in the counts_df to group samples by before calculating the correlation matrix (e.g., "source_name" or other metadata variable).

    Returns
    -------
    pearson_corrs: list
        A list of DataFrames representing the gene-gene correlation matrices.
    grouped_data: list
        A list of DataFrames representing the grouped count data used for correlation calculation.
    """
    pearson_corrs = []
    grouped_data = []
    for group in counts_df[group_by].unique():
        # Select data from the group and only numeric columns (gene counts)
        group_data = counts_df[counts_df[group_by] == group].select_dtypes(
            include="number"
        )
        grouped_data.append(group_data)
        # Correlation matrix - Pearson correlation
        pearson_corr = group_data.corr(method="pearson")
        pearson_corrs.append(pearson_corr)

    return pearson_corrs, grouped_data


def lioness_ppi(
    grouped_count_df: list,
    corr_matrices: list,
    ppi_network: pd.DataFrame,
    threshold=0.25,
    verbose=True,
) -> list:
    """
    Function that applies the LIONES algorithm to the gene co-expression matrices with PPI network as prior to infer sample-specific gene-gene interaction networks.

    Parameters
    ----------
    grouped_count_df: list
        A list of DataFrames representing the grouped count data used for correlation calculation.
    corr_matrices: list
        A list of DataFrames representing the gene-gene correlation matrices for each group.
    ppi_network: pd.DataFrame
        A DataFrame containing the PPI network with columns for the interacting proteins.
    threshold: float
        The correlation threshold to consider an edge as present in the inferred network (default is 0.25).
    verbose: bool
        Whether to print progress information during the LIONESS algorithm execution (default is True).

    Returns
    -------
    list
        A list of DataFrames representing the inferred sample-specific gene-gene interaction networks.
    """
    sample_networks = {}
    I = len(grouped_count_df)

    for i, group_data in enumerate(grouped_count_df):
        # Relevant data
        samples = group_data.index.tolist()
        genes = group_data.columns.tolist()
        N = group_data.shape[0]

        # Correlation matrix
        corr_matrix = corr_matrices[i]

        # Mapping from gene to index
        gene_to_idx = {gene: i for i, gene in enumerate(genes)}

        # Get PPI edges
        ppi_edges = get_ppi_edges(ppi_network)

        # Get indices for the edges that are in the PPI network
        ppi_indices = []
        ppi_gene_pairs = []
        for gene_a, gene_b in ppi_edges:
            if gene_a in gene_to_idx and gene_b in gene_to_idx:
                idx_a = gene_to_idx[gene_a]
                idx_b = gene_to_idx[gene_b]
                # Ensure i < j
                ppi_indices.append((min(idx_a, idx_b), max(idx_a, idx_b)))
                ppi_gene_pairs.append((gene_a, gene_b))

        # Fisher z-transform of G
        G = corr_matrix.values
        Z = np.arctanh(np.clip(G, -1 + 1e-12, 1 - 1e-12))

        # Iterate on all samples, leave-one-out G_{-s} correlation and LIONESS in z-space
        for s in tqdm(
            samples,
            desc=f"LIONESS samples (PPI filtered) for group {i}/{I}",
            disable=not verbose,
        ):
            # Drop samples (s)
            data_minus_s = group_data.drop(index=s)

            # Correlation on N-1 samples
            G_minus_s = data_minus_s.corr(method="pearson").values

            # Z-space
            Z_minus_s = np.arctanh(np.clip(G_minus_s, -1 + 1e-12, 1 - 1e-12))

            # LIONESS in z-space
            Z_s = N * (Z - Z_minus_s) + Z_minus_s

            # PPI filtering_ extract weights for just the edges present in the PPI network
            Z_s_ppi = np.array([Z_s[i, j] for i, j in ppi_indices])

            # Going back to R space
            G_s_ppi = np.tanh(Z_s_ppi)

            # Filter correlations under threshold
            keep_mask = np.abs(G_s_ppi) >= threshold

            # Create DataFrame for the sample-specific network
            kept_pairs = [ppi_gene_pairs[i] for i in np.where(keep_mask)[0]]
            kept_weights = G_s_ppi[keep_mask]
            edges = pd.DataFrame(
                {
                    "geneA": [p[0] for p in kept_pairs],
                    "geneB": [p[1] for p in kept_pairs],
                    "weight": kept_weights,
                }
            )
            # Save
            sample_networks[s] = edges.reset_index(drop=True)

        return sample_networks


def gene_networks_to_pyg(
    sample_networks: list,
    joint_data: pd.DataFrame,
    gene_embeddings: torch.Tensor,
    bio_col="source_name",
):
    """
    Function that converts sample-specific gene networks into PyG objects for use in graph neural network models.

    Parameters
    ----------
    sample_networks: list
        A list of DataFrames representing the networks. Each DataFrame should have columns "geneA", "geneB", and "weight" representing the edges and their weights.
    joint_data: pd.DataFrame
        A DataFrame containing the merged count data and sample metadata, with samples as rows and genes as columns.
    gene_embeddings: torch.Tensor
        A tensor containing pre-trained gene embeddings, where the order of genes corresponds to the columns in joint_data.
    bio_col: str
        The column name in joint_data that contains the biological phenotypes.

    Returns
    -------
    list
        A list of PyG Data objects representing the sample-specific gene networks, ready for use in graph neural network models.
    """
    # Get only gene expression data (numeric columns)
    norm_data = joint_data.select_dtypes(include="number")
    # Getting gene list
    genes = norm_data.columns.tolist()
    # Getting gene to index mapping
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    # Getting biological phenotypes as dictionary per sample
    bio_to_source = joint_data[bio_col].to_dict()
    # Getting mapping for biological phenotypes to integer labels
    bio_map = {lab: i for i, lab in enumerate(sorted(joint_data[bio_col].unique()))}

    data_list = []
    static_embeddings = (
        gene_embeddings.cpu()
    )  # Ensure gene embeddings are on CPU for PyG

    for s, df in tqdm(
        sample_networks.items(),
        desc="Building PyG Data objects for sample-specific networks",
    ):

        if s not in norm_data.index:
            continue  # Skip samples that are not in the normalized data

        # 1. Node Features: [Expression (1) + Gene embeddings (hidden_dim)]
        expr = norm_data.loc[s]
        expr_tensor = torch.from_numpy(expr)
        x = torch.cat([expr_tensor, static_embeddings], dim=1)

        # 2. Process positive and negative edges
        edges = []
        edge_types = []
        edge_weights = []

        for _, row in df.iterrows():

            a, b, w = row["geneA"], row["geneB"], float(row["weight"])
            if a not in gene_to_idx or b not in gene_to_idx:
                continue  # Skip edges where genes are not in the gene list

            i, j = gene_to_idx[a], gene_to_idx[b]
            w_abs = abs(w)
            e_type = 0 if w > 0 else 1  # 0 for positive, 1 for negative

            edges.extend([(i, j), (j, i)])  # Add both directions for undirected graph
            edge_types.extend([e_type, e_type])  # Same type for both directions
            edge_weights.extend([w_abs, w_abs])  # Same weight for both directions

        if not edges:
            continue  # Skip samples with no edges after filtering

        # 3. Create PyG Data object and assign labels
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
        )  # Shape [2, num_edges]
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)  # Shape [num_edges]
        edge_type = torch.tensor(edge_types, dtype=torch.long)  # Shape [num_edges]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.edge_type = edge_type
        data.sample_name = s

        # 4. Assign label based on biological phenotype
        src = bio_to_source[s]
        data.y = torch.tensor(bio_map[src], dtype=torch.long)

        data_list.append(data)

    return data_list
