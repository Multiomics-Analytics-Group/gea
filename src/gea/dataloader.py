# gea/dataloader.py
from copyreg import pickle
import pandas as pd
import requests
import io
from transformers import BertModel
from huggingface_hub import hf_hub_download
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


def load_counts(path: str, delim="\t", index_col="Geneid") -> pd.DataFrame:
    """
    Function used to load inital count matrix data as a pd.DataFrame.

    Parameters
    ----------
    path: str
        Path to the count matrix file. The file should be in a format that can be read by pandas (e.g., CSV, TSV, Excel).
    delim: str
        The delimiter used in the count matrix file.
    index_col: str
        The column name to use as the row labels.

    Returns
    -------
    pd.DataFrame
        The loaded count matrix as a pandas DataFrame.
    """
    return pd.read_csv(path, sep=delim, index_col=index_col)


def load_network():
    pass


def load_pubchem_network():
    pass


def load_string_ppi_network(
    gene_list: list,
    species: int = 9606,
    conf_score: int = 600,
    api_url: str = "https://string-db.org/api",
    ensembl_to_symbol: dict = None,
) -> pd.DataFrame:
    """
    Extract a protein-protein interaction network from STRING for a list of genes.

    Parameters
    ----------
    gene_list : list
        Gene identifiers to query. Can be gene symbols (default) OR Ensembl gene
        IDs when ensembl_to_symbol is provided.
    species : int
        NCBI taxonomy ID (default 9606 = human).
    conf_score : int
        Minimum combined confidence score (0–1000). Default 600.
    api_url : str
        STRING API base URL.
    ensembl_to_symbol : dict, optional
        Mapping from Ensembl gene ID → gene symbol, as returned by
        filter_protein_coding. When provided, gene_list is assumed to contain
        Ensembl IDs: they are translated to symbols for the STRING query, and
        the returned DataFrame is enriched with 'ensemblId_A' / 'ensemblId_B'
        columns so that downstream steps can work with Ensembl IDs directly.

    Returns
    -------
    pd.DataFrame
        STRING network with columns including preferredName_A, preferredName_B,
        score. If ensembl_to_symbol was given, also includes ensemblId_A and
        ensemblId_B columns.
    """
    if ensembl_to_symbol is not None:
        # Build reverse map and translate to symbols for the query
        symbol_to_ensembl = {v: k for k, v in ensembl_to_symbol.items()}
        query_symbols = [
            ensembl_to_symbol[g] for g in gene_list if g in ensembl_to_symbol
        ]
    else:
        query_symbols = gene_list
        symbol_to_ensembl = None

    request_url = "/".join([api_url, "tsv", "network"])
    params = {
        "identifiers": "\n".join(query_symbols),
        "species": species,
        "required_score": conf_score,
        "caller_identity": "script",
    }

    try:
        response = requests.post(request_url, data=params)
        response.raise_for_status()
        ppi_network = pd.read_csv(io.StringIO(response.text), sep="\t")

        print("Successfully retrieved PPI network!")
        print(f"Found {len(ppi_network)} interactions.")

        # Add Ensembl ID columns so filter_ppi_nodes can use them
        if symbol_to_ensembl is not None:
            ppi_network["ensemblId_A"] = ppi_network["preferredName_A"].map(
                symbol_to_ensembl
            )
            ppi_network["ensemblId_B"] = ppi_network["preferredName_B"].map(
                symbol_to_ensembl
            )

        return ppi_network

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")


def load_metadata(path: str, delim=",", index_col=None) -> pd.DataFrame:
    """
    Function used to load metadata as a pd.DataFrame.

    Parameters
    ----------
    path: str
        Path to the metadata file. The file should be in a format that can be read by pandas (e.g., CSV, TSV, TXT).
    delim: str
        The delimiter used in the metadata file (default is comma).
    index_col: str
        The column name to use as the row labels (default is None).

    Returns
    -------
    pd.DataFrame
        The loaded metadata as a pandas DataFrame.
    """
    return pd.read_csv(path, sep=delim, index_col=index_col)


def load_geneformer(
    model_name="ctheodoris/Geneformer",
    filename="token_dictionary_gc104M.pkl",
    subfolder="geneformer",
):
    """
    Function used to load the Geneformer model and its token dictionary.

    Parameters
    ----------
    model: str
        The Hugging Face model identifier for the Geneformer model (default is "ctheodoris/Geneformer").
    filename: str
        The name of the token dictionary file on Hugging Face (default is "token_dictionary_gc104M.pkl").
    subfolder: str
        The subfolder where the token dictionary is on Hugging Face (default is "geneformer").

    Returns
    -------
    BertModel
        The loaded Geneformer model.
    dict
        The loaded token dictionary.
    """
    # Load Geneformer model
    try:
        # Load model
        model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval()

        # Load vocabulary
        dict_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            subfolder=subfolder,
        )
        with open(dict_path, "rb") as f:
            vocab = pickle.load(f)

        return model, vocab

    except Exception as e:
        print(f"Error loading Geneformer model or token dictionary: {e}")
        return None, None


class EmbeddingDataset(Dataset):
    """
    Standardized embedding dataset format.

    Reads embeddings, annotations, entities, predictions, and targets from a
    NumPy `.npz` file.

    The `.npz` file must contain:

        - "embeddings": A NumPy array of shape (N, D), where N is the number
          of samples (nodes, edges or graphs) and D is the embedding dimension.

        - "annotations": A NumPy object array containing a dictionary of
          annotations for each sample.

        - "entities": A NumPy array containing the identifier of the entity
          associated with each sample (e.g., SMILES string for molecular data).

        - "prediction": A NumPy array containing the model prediction associated
          with each sample.

        - "target": A NumPy array containing the ground-truth target associated
          with each sample.

    Sample correspondence:
        embeddings[i]
        annotations[i]
        entities[i]
        prediction[i]
        target[i]

        all correspond to the same sample.

    Embeddings are representation-agnostic and can originate from any source
    (node-level, graph-level, bond-level, sequence-level, etc.) as long as
    they are converted into fixed-size vectors.

    The dataset returns samples in the format:
        {
            "embedding": Tensor[D],
            "annotation": {
                annotation_name: Tensor[...]
            },
            "entity": str,
            "prediction": Tensor,
            "target": Tensor
        }
    """

    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)

        self.embeddings = torch.tensor(
            data["embeddings"],
            dtype=torch.float32
        )

        self.annotations = data["annotations"]
        self.entities = data["entities"]
        self.predictions = data["prediction"]
        self.targets = data["target"]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):

        annotation = {
            label: torch.tensor(value, dtype=torch.float32)
            for label, value in self.annotations[idx].items()
        }

        return {
            "embedding": self.embeddings[idx],
            "annotation": annotation,
            "entity": self.entities[idx],
            "prediction": self.predictions[idx],
            "target": self.targets[idx],
        }


def load_embedding_metadata(npz_file) -> pd.DataFrame:
    """
    Read the identity of every embedding in an ``.npz`` file as a DataFrame,
    without loading the embedding matrix into a Dataset.

    Because the loader used by ``gea.gea.extract_embeddings`` is unshuffled, row
    *i* of this table describes row *i* of the embedding matrix — and therefore
    row *i* of any SAE activation matrix computed from it. This is the table to
    join feature activations onto in order to say *which gene, which gene pair
    (and with which sign), in which graph* a feature fired on.

    Parameters
    ----------
    npz_file : str or pathlib.Path
        File written by ``gea.gea.save_embeddings`` / ``export_embeddings``.

    Returns
    -------
    pd.DataFrame
        One row per embedding. Identity columns (``graph_id``, and ``gene`` or
        ``gene_a``/``gene_b`` for node and edge levels, plus ``disease`` and
        ``cell_type`` when label names were given), the scalar annotations
        (including the signed edge ``weight``), and ``entity``, ``prediction``
        and ``target``.

        The one-hot group annotations are skipped: they say the same thing as the
        ``disease`` / ``cell_type`` string columns, which are easier to group by.
    """
    data = np.load(npz_file, allow_pickle=True)

    id_cols = [
        c
        for c in ("graph_id", "gene", "gene_a", "gene_b", "disease", "cell_type")
        if c in data.files
    ]
    df = pd.DataFrame({c: data[c] for c in id_cols})

    annotations = data["annotations"]
    if len(annotations):
        for label, value in annotations[0].items():
            if np.ndim(value) == 0:  # skip one-hot vectors
                df[label] = [a[label] for a in annotations]

    df["entity"] = data["entities"]
    prediction = data["prediction"]
    if prediction.ndim == 1:
        df["prediction"] = prediction
    df["target"] = data["target"]

    return df