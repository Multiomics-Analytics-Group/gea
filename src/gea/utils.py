# gea/utils.py
import numpy as np
import pandas as pd
import mygene
from transformers import BertModel
import torch
from tqdm import tqdm
import os
from typing import Tuple, Dict


def filter_protein_coding(
    gene_data: pd.DataFrame, species: str = "human"
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Filter a gene expression DataFrame to retain only protein-coding genes,
    keeping Ensembl gene IDs as the index throughout.

    Unlike ensembl_to_gene, this function does NOT convert to gene symbols —
    Ensembl IDs remain the primary identifier so that ESM-2 embeddings (which
    are indexed by Ensembl ID) can be looked up directly.

    Parameters
    ----------
    gene_data : pd.DataFrame
        Raw count matrix with Ensembl gene IDs (ENSG*) as index.
    species : str
        Species string passed to mygene (e.g. "human", "mouse").

    Returns
    -------
    filtered : pd.DataFrame
        Subset of gene_data containing only protein-coding genes,
        with Ensembl IDs retained as index.
    ensembl_to_symbol : dict
        Mapping from Ensembl gene ID → HGNC gene symbol, useful for
        labelling nodes in visualisations and querying STRING.
    """
    ensembl_ids = [i for i in gene_data.index if str(i).startswith("ENSG")]
    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol,type_of_gene",
        species=species,
        as_dataframe=True,
    )
    gene_info = gene_info[~gene_info.index.duplicated(keep="first")]

    type_cols = [
        c for c in gene_info.columns if "type" in c.lower() or "biotype" in c.lower()
    ]
    if type_cols:
        is_protein = (
            gene_info[type_cols[0]].astype(str).str.lower().str.contains("protein")
        )
        gene_info = gene_info[is_protein]

    valid_ensembl = set(gene_info.index)
    filtered = gene_data.loc[gene_data.index.isin(valid_ensembl)].copy()

    ensembl_to_symbol = (
        gene_info["symbol"].dropna().loc[gene_info.index.isin(valid_ensembl)].to_dict()
    )

    print(
        f"Retained {len(filtered)} protein-coding genes "
        f"(from {len(ensembl_ids)} Ensembl IDs queried)."
    )
    return filtered, ensembl_to_symbol


def load_esm_embeddings(
    esm_file: str, ensembl_ids: list, aggregation: str = "mean"
) -> torch.Tensor:
    """
    Load pre-computed ESM-2 embeddings from a saved file, producing one
    embedding vector per gene by aggregating over isoforms.

    Expected file format (produced by scripts/get_esm_embeddings.py):
        { ensembl_id: { transcript_id: tensor([hidden_dim]) } }

    Genes absent from the file receive a random Gaussian embedding so that
    downstream processing is never blocked by missing entries.

    Parameters
    ----------
    esm_file : str
        Path to the .pt file saved by get_esm_embeddings.py.
    ensembl_ids : list of str
        Ordered list of Ensembl gene IDs matching the row order of the
        normalised counts DataFrame.
    aggregation : str
        How to collapse isoform embeddings: 'mean' (default) or 'max'.

    Returns
    -------
    torch.Tensor, shape [len(ensembl_ids), hidden_dim]
    """
    nested = torch.load(esm_file, weights_only=False)

    # Infer embedding dimension from the first stored entry
    first_iso = next(iter(next(iter(nested.values())).values()))
    hidden_dim = first_iso.shape[0]

    embeddings = torch.zeros(len(ensembl_ids), hidden_dim)
    found = 0

    for i, gene_id in enumerate(tqdm(ensembl_ids, desc="Loading ESM-2 embeddings")):
        if gene_id in nested:
            isoform_tensors = torch.stack(list(nested[gene_id].values()))
            if aggregation == "mean":
                embeddings[i] = isoform_tensors.mean(dim=0)
            else:
                embeddings[i] = isoform_tensors.max(dim=0).values
            found += 1
        else:
            embeddings[i] = torch.randn(hidden_dim)

    print(f"Found ESM-2 embeddings for {found}/{len(ensembl_ids)} genes.")

    # L2-normalise so every gene embedding has unit norm.
    # ESM-2 CLS tokens vary widely in magnitude; without this the embedding
    # directions dominate the gradient and the single expression scalar is
    # effectively invisible to the first GNN layer.
    norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    embeddings = embeddings / norms

    return embeddings


def compress_embeddings_pca(
    embeddings: torch.Tensor,
    n_components: int = 256,
    whiten: bool = True,
) -> torch.Tensor:
    """
    Reduce ESM-2 (or any) embedding matrix with PCA, then L2-renormalise.

    Using offline PCA instead of a learned linear projection is preferable for
    small datasets (~300 graphs): PCA converges on the first call and introduces
    zero gradient burden, matching the effective dimensionality of Geneformer
    (256) without requiring the GNN to simultaneously learn the compression.

    Parameters
    ----------
    embeddings : torch.Tensor, shape [n_genes, in_dim]
        L2-normalised gene embeddings (output of load_esm_embeddings).
    n_components : int
        Target dimensionality (default: 256, matching Geneformer).
    whiten : bool
        If True (default), divide by sqrt of eigenvalue so all components have
        unit variance — makes the downstream scale uniform regardless of PCA
        ordering.

    Returns
    -------
    torch.Tensor, shape [n_genes, n_components]
        PCA-compressed, L2-renormalised embeddings.
    """
    from sklearn.decomposition import PCA

    X = embeddings.float().numpy()
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    X_pca = pca.fit_transform(X).astype(np.float32)

    result = torch.from_numpy(X_pca)
    norms = result.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    result = result / norms

    explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA: {embeddings.shape[1]}→{n_components} dims, "
        f"{explained:.1%} variance retained."
    )
    return result


def query_biomart(attributes, attempts=10):

    from pybiomart import Server

    for attempt in range(attempts):

        try:
            server = Server(host="http://www.ensembl.org", use_cache=False)
            dataset = server["ENSEMBL_MART_ENSEMBL"]["hsapiens_gene_ensembl"]
            return dataset.query(attributes=attributes)

        except Exception as e:
            print(f"Attempt {attempt+1}/{attempts} failed: {e}")

    raise RuntimeError(f"All {attempts} attempts failed.")


def ensembl_to_gene(gene_data: pd.DataFrame, species="human") -> pd.DataFrame:
    """
    Function used to convert Ensembl gene IDs to gene symbols.

    Parameters
    ----------
    gene_data: pd.DataFrame
        A DataFrame containing Ensembl gene IDs as the index.
    species: str
        The species for which to query gene information (default is "human").

    Returns
    -------
    pd.DataFrame
        A DataFrame with gene symbols as the index.
    """
    ensembl_ids = gene_data.index.tolist()
    # Sanity check: Only keep Ensembl IDs that start with "ENSG"
    ensembl_ids = [i for i in ensembl_ids if i.startswith("ENSG")]
    # Query MyGeneInfo to get gene symbols
    mg = mygene.MyGeneInfo()

    # Query MyGeneInfo to get gene symbols and other information
    fields = "symbol, type_of_gene, entrezgene, ensembl"
    gene_info = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields=fields,
        species=species,
        as_dataframe=True,
    )
    gene_info = gene_info[~gene_info.index.duplicated(keep="first")]

    # Find column that have gene-type/biotype information
    type_cols = [
        c for c in gene_info.columns if "type" in c.lower() or "biotype" in c.lower()
    ]

    if type_cols:
        tcol = type_cols[0]
        # Filter for protein-coding genes
        is_protein = gene_info[tcol].astype(str).str.lower().str.contains("protein")
        gene_info = gene_info[is_protein]

    # Insert gene symbols into the original gene_data DataFrame
    gene_data.insert(0, "gene_symbol", gene_info["symbol"].reindex(gene_data.index))
    gene_data = gene_data.dropna(subset=["gene_symbol"]).set_index("gene_symbol")

    return gene_data


def get_gene_list(gene_data: pd.DataFrame) -> list:
    """
    Function used to get a list of gene symbols from a DataFrame.

    Parameters
    ----------
    gene_data: pd.DataFrame
        A DataFrame containing gene symbols as the index.

    Returns
    -------
    list
        A list of gene symbols.
    """
    return gene_data.index.tolist()


def get_ppi_edges(ppi_df: pd.DataFrame) -> set:
    """
    Return a set of (id_a, id_b) tuples representing PPI edges.

    Uses Ensembl ID columns ('ensemblId_A' / 'ensemblId_B') when present —
    as added by load_string_ppi_network when called with ensembl_to_symbol —
    and falls back to gene-symbol columns ('preferredName_A' / 'preferredName_B')
    otherwise. Rows with NaN identifiers are skipped.
    """
    if "ensemblId_A" in ppi_df.columns and "ensemblId_B" in ppi_df.columns:
        col_a, col_b = "ensemblId_A", "ensemblId_B"
    else:
        col_a, col_b = "preferredName_A", "preferredName_B"

    ppi_edges = set()
    for _, row in ppi_df.iterrows():
        a, b = row[col_a], row[col_b]
        if pd.notna(a) and pd.notna(b):
            ppi_edges.add(tuple(sorted((a, b))))

    return ppi_edges


def get_geneformer_embeddings(
    model: BertModel, vocab: dict, gene_list: list
) -> pd.DataFrame:
    """
    Function used to get gene embeddings from a Geneformer model.

    Parameters
    ----------
    model: BertModel
        A pre-trained Geneformer model.
    vocab: dict
        The token dictionary.
    gene_list: list
        A list of gene symbols for which to extract embeddings.

    Returns
    -------
    torch.Tensor
        A tensor containing the gene embeddings.
    """
    # Mapping symbols to Ensembl IDs
    mapping = query_biomart(attributes=["hgnc_symbol", "ensembl_gene_id"])
    mapping = mapping.rename(
        columns={"HGNC symbol": "symbol", "Gene stable ID": "ensembl_id"}
    ).dropna()
    symbol_to_ensembl = dict(zip(mapping["symbol"], mapping["ensembl_id"]))

    # Extract embeddings for genes in the gene list
    embedding_matrix = model.embeddings.word_embeddings.weight
    hidden_dim = embedding_matrix.shape[1]

    # Create tensor [num_genes, hidden_dim]
    gene_embeddings = torch.zeros((len(gene_list), hidden_dim))

    found = 0

    for i, gene in enumerate(tqdm(gene_list, desc="Extracting Geneformer embeddings")):
        ensembl_id = symbol_to_ensembl.get(gene)
        if ensembl_id and ensembl_id in vocab:
            token_id = vocab[ensembl_id]
            if (
                token_id < embedding_matrix.shape[0]
            ):  # Check if token_id is within bounds
                gene_embeddings[i] = embedding_matrix[token_id]
                found += 1
            else:
                gene_embeddings[i] = torch.randn(
                    hidden_dim
                )  # Random embedding for out-of-bounds token_id
        else:
            gene_embeddings[i] = torch.randn(
                hidden_dim
            )  # Random embedding for missing gene

    print(f"Found embeddings for {found}/{len(gene_list)} genes.")

    return gene_embeddings
