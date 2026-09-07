import argparse
import pickle
import numpy as np
from gea_molecules.annotation_funcs import (
    annotate_node_GroverEmbeds,
    annotate_graph_GroverEmbeds,
)

def main(args):

    with open(args.motif_dict, "rb") as f:
        motif_dict = pickle.load(f)

    # Node embeddings
    node_obj = annotate_node_GroverEmbeds(
        args.embeddings_path,
        motif_dict=motif_dict,
        embedding_key=args.embedding_key_node
    )

    np.savez_compressed(
        args.node_output,
        embeddings=node_obj["embeddings"],
        annotations=np.array(node_obj["annotations"], dtype=object),
        entities=np.array(node_obj["entities"], dtype=object),
        prediction=node_obj["prediction"],
        target=node_obj["target"],
    )

    # Graph embeddings
    graph_obj = annotate_graph_GroverEmbeds(
        args.embeddings_path,
        embedding_key=args.embedding_key_graph
    )

    np.savez_compressed(
        args.graph_output,
        embeddings=graph_obj["embeddings"],
        annotations=np.array(graph_obj["annotations"], dtype=object),
        entities=np.array(graph_obj["entities"], dtype=object),
        prediction=graph_obj["prediction"],
        target=graph_obj["target"],
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Annotate Grover molecular embeddings."
    )

    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to Grover embeddings .pt file"
    )

    parser.add_argument(
        "--motif_dict",
        type=str,
        default='dict/motif_dictionary.pkl',
        help="Path to motif dictionary .pkl file"
    )

    parser.add_argument(
        "--node_output",
        type=str,
        default="node_embeddings.npz",
        help="Output path for node embeddings"
    )

    parser.add_argument(
        "--graph_output",
        type=str,
        default="graph_embeddings.npz",
        help="Output path for graph embeddings"
    )

    parser.add_argument(
        "--embedding_key_node",
        type=str,
        default="atom_from_atom",
        help="Key for node embeddings in the Grover embeddings dictionary"
    )

    parser.add_argument(
        "--embedding_key_graph",
        type=str,
        default="graph_from_atom_from_atom",
        help="Key for graph embeddings in the Grover embeddings dictionary"
    )

    args = parser.parse_args()

    main(args)
