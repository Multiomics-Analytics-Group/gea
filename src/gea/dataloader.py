# gea/dataloader.py
from copyreg import pickle
import pandas as pd
import requests
import io
from transformers import BertModel
from huggingface_hub import hf_hub_download
import pickle


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


def load_ppi_network(
    gene_list: list, species=9606, conf_score=600, api_url="https://string-db.org/api"
) -> pd.DataFrame:
    """
    Function used to extract a protein-protein interaction (PPI) network from the STRING database for a given list of genes.

    Parameters
    ----------
    gene_list: list
        A list of gene symbols for which to extract the PPI network.
    species: int
        The NCBI taxonomy identifier for the species of interest (default is 9606 for human).
    conf_score: int
        The confidence score threshold for including interactions (default is 600, highest is 900).
    api_url: str
        The base URL for the STRING database API (default is "https://string-db.org/api").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the PPI network with columns for the interacting proteins and their confidence scores.
    """
    # API method to get network
    method = "network"
    # Format list of genes into single string
    id_string = "\n".join(gene_list)
    # Request URL construct
    request_url = "/".join([api_url, "tsv", method])

    # Parameters for the API call
    params = {
        "identifiers": id_string,
        "species": species,
        "required_score": conf_score,
        "caller_identity": "script",
    }

    # Making API call
    try:
        response = requests.post(request_url, data=params)
        response.raise_for_status()

        # io.StringIO function treats the response text as file
        ppi_network = pd.read_csv(io.StringIO(response.text), sep="\t")

        print("Successfully retrieved PPI network!")
        print(f"Found {len(ppi_network)} interactions.")
        print(ppi_network.head())

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
