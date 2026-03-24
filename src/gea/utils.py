# gea/utils.py
import pandas as pd
import numpy as np
import mygene
from transformers import BertModel
import torch
from pybiomart import Server
from tqdm import tqdm


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


def get_ppi_edges(ppi_df: pd.DataFrame) -> list:
    """
    Function used to get a list of gene symbols from a DataFrame.

    Parameters
    ----------
    ppi_df: pd.DataFrame
        A DataFrame containing PPI information with columns "preferredName_A" and "preferredName_B".

    Returns
    -------
    list
        A list of tuples representing the edges in the PPI network.
    """
    ppi_edges = set()
    for _, row in ppi_df.iterrows():
        edge = tuple(sorted((row["preferredName_A"], row["preferredName_B"])))
        ppi_edges.add(edge)

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
    # Via 1: Ensembl servers
    biomart_urls = [
        "http://www.ensembl.org",  # Original
        "http://useast.ensembl.org",  # US East mirror
        "http://asia.ensembl.org",  # Asia mirror
    ]

    mapping_successful = False

    for url in biomart_urls:
        try:
            print(f"Attemping to fetch gene mapping from Ensembl at: {url}...")
            server = Server(host=url)
            dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets[
                "hsapiens_gene_ensembl"
            ]
            mapping = dataset.query(attributes=["hgnc_symbol", "ensembl_gene_id"])

            mapping = mapping.rename(
                columns={"HGNC symbol": "symbol", "Gene stable ID": "ensembl_id"}
            ).dropna()
            symbol_to_ensembl = dict(zip(mapping["symbol"], mapping["ensembl_id"]))
            print(f"Succesfully mapped genes using BioMart ({url}).")
            mapping_successful = True
            break

        except Exception as e:
            print(f"BioMart query failed for {url}.")

    if not mapping_successful:
        print(
            "All Ensembl BioMart servers are currently down. Falling back to MyGene.info API..."
        )
        try:
            mg = mygene.MyGeneInfo()
            results = mg.querymany(
                gene_list,
                scopes="symbol",
                fields="human",
                as_dataframe=True,
                verbose=False,
            )

            if "ensembl.gene" in results.columns:
                valid_results = results.dropna(subset=["ensembl.gene"])

                for symbol, _ in valid_results.iterrows():
                    ensembl_id = ["ensembl.gene"]
                    if isinstance(ensembl_id, list):
                        ensembl_id = ensembl_id[0]

                    symbol_to_ensembl[symbol] = ensembl_id

                print("Succesfully mapped genes using MyGene.")

            else:
                print("Error: 'ensembl.gene' field not found in MyGene response.")

        except Exception as e:
            print(f"Critical Error: MyGene fallback also failed: {e}")

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
