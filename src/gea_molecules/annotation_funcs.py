import torch
from rdkit import Chem
from motifAnnotation import motifAnnotation
import numpy as np

def annotate_node_GroverEmbeds(embeddings_path, motif_dict):
    embeddings = torch.load(embeddings_path)

    emb_array = []
    annotations = []
    entities = []
    predictions = []
    targets = []

    for smi, emb_dict in embeddings.items():

        mol = Chem.MolFromSmiles(smi)

        annotation = motifAnnotation(
            mol=mol, 
            motif_dict=motif_dict
        )

        node_embeds = emb_dict["atom_from_atom"] # We need to see if we want to consider atom_from_bond

        emb_array.append(node_embeds.cpu().numpy().astype(np.float16))

        n_nodes = node_embeds.shape[0]

        for i in range(n_nodes):
            annotations.append({
                name: values[i]
                for name, values in annotation.items()
            })
            entities.append(smi)
            predictions.append(emb_dict["prediction"])
            targets.append(emb_dict["target"])

    return {
        "embeddings": np.concatenate(emb_array, axis=0),
        "annotations": annotations,
        "entities": entities,
        "prediction": np.array(predictions),
        "target": np.array(targets),
    }

def annotate_graph_GroverEmbeds(embeddings_path, motif_dict):
    embeddings = torch.load(embeddings_path)

    emb_array = []
    annotations = []
    entities = []
    predictions = []
    targets = []

    for smi, emb_dict in embeddings.items():

        mol = Chem.MolFromSmiles(smi)
        annotation = motifAnnotation(
            mol=mol,
            motif_dict=motif_dict
        )

        node_embeds = emb_dict["atom_from_atom"]

        # node embeddings -> graph embedding
        graph_embed = node_embeds.mean(dim=0)

        emb_array.append(
            graph_embed.cpu().numpy().astype(np.float16)
        )

        annotations.append({
            key: sum(value)
            for key, value in annotation.items()
        })

        entities.append(smi)
        predictions.append(emb_dict["prediction"])
        targets.append(emb_dict["target"])

    return {
        "embeddings": np.stack(emb_array, axis=0),
        "annotations": annotations,
        "entities": entities,
        "prediction": np.array(predictions),
        "target": np.array(targets),
    }

def motifAnnotation(mol, motif_dict):
    mol_dict = {}

    for motif, smarts in motif_dict.items():
        patt = Chem.MolFromSmarts(smarts)
        mask = [0]*mol.GetNumAtoms()
        for match in mol.GetSubstructMatches(patt):
            for atom_idx in match:
                mask[atom_idx] = 1
    
        mol_dict[motif] = mask

    return mol_dict