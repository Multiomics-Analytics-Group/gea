import torch
from rdkit import Chem
import numpy as np
from rdkit.Chem import Fragments, Descriptors, Crippen, Lipinski, rdMolDescriptors
import inspect
import pandas as pd

def get_motif_matches(mol, motif_dict):
    matches = {}

    for name, smarts in motif_dict.items():
        pattern = Chem.MolFromSmarts(smarts)

        if pattern is None:
            matches[name] = []
            continue

        matches[name] = mol.GetSubstructMatches(pattern)

    return matches

def motif_annotation(mol, motif_dict, level="node"):

    matches = get_motif_matches(mol, motif_dict)

    if level == "node":

        annotations = {}
        n_atoms = mol.GetNumAtoms()

        for name, motif_matches in matches.items():

            mask = [0] * n_atoms

            for match in motif_matches:
                for atom_idx in match:
                    mask[atom_idx] = 1

            annotations[name] = mask

    elif level == "graph":

        annotations = {
            name: len(motif_matches)
            for name, motif_matches in matches.items()
        }

    else:
        raise ValueError(
            "`level` must be either 'node' or 'graph'"
        )

    return annotations


def calculate_descriptors(mol):

    if mol is None:
        return None

    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "HeavyAtoms": Lipinski.HeavyAtomCount(mol),
        "Rings": Lipinski.RingCount(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "MolarRefractivity": Crippen.MolMR(mol),
    }

def filter_constant_annotations(annotations):
    """
    Remove annotations that are constant across all samples.

    Parameters
    ----------
    annotations : list of dict
        One annotation dictionary per sample.

    Returns
    -------
    annotations : list of dict
        Filtered annotation dictionaries.
    retained_annotations : list of str
        Names of annotations that vary across samples.
    """
    if not annotations:
        return annotations, []

    annotation_names = annotations[0].keys()

    retained_annotations = [
        name
        for name in annotation_names
        if len({sample[name] for sample in annotations}) > 1
    ]

    filtered_annotations = [
        {
            name: sample[name]
            for name in retained_annotations
        }
        for sample in annotations
    ]

    removed = set(annotation_names) - set(retained_annotations)

    print(
        f"Removed {len(removed)} constant annotations; "
        f"retained {len(retained_annotations)}."
    )

    return filtered_annotations, retained_annotations

def annotate_node_GroverEmbeds(embeddings_path, motif_dict, embedding_key="atom_from_atom"):

    embeddings = torch.load(embeddings_path)

    emb_array = []
    annotations = []
    entities = []
    predictions = []
    targets = []

    for smi, emb_dict in embeddings.items():

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            print(f"Warning: Could not parse SMILES '{smi}'. Skipping.")
            continue

        annotation = motif_annotation(
            mol=mol, 
            motif_dict=motif_dict,
            level = "node"
        )

        node_embeds = emb_dict[embedding_key] # We need to see if we want to consider atom_from_bond

        emb_array.append(node_embeds.cpu().numpy().astype(np.float16))

        n_nodes = node_embeds.shape[0]

        if n_nodes != mol.GetNumAtoms():
            raise ValueError(
                f"Number of nodes ({n_nodes}) does not match "
                f"number of atoms ({mol.GetNumAtoms()}) for {smi}"
            )

        for i in range(n_nodes):
            annotations.append({
                name: values[i]
                for name, values in annotation.items()
            })
            
            entities.append(smi)
            predictions.append(emb_dict["prediction"])
            targets.append(emb_dict["target"])

    annotations, retained_annotations = filter_constant_annotations(
        annotations
    )

    return {
        "embeddings": np.concatenate(emb_array, axis=0),
        "annotations": annotations,
        "entities": entities,
        "prediction": np.array(predictions),
        "target": np.array(targets),
        "retained_annotations": retained_annotations,
    }

def annotate_graph_GroverEmbeds(
        embeddings_path, 
        motif_dict, 
        embedding_key = "graph_from_atom_from_atom",
        annotations_csv=None,):

    embeddings = torch.load(embeddings_path)

    # Load optional additional annotations
    extra_annotations = {}

    if annotations_csv is not None:
        df = pd.read_csv(annotations_csv)

        if df.shape[1] < 2:
            raise ValueError(
                "The annotations CSV must contain a SMILES column "
                "and at least one annotation column."
            )

        smiles_col = df.columns[0]

        if df[smiles_col].duplicated().any():
            duplicates = df.loc[
                df[smiles_col].duplicated(), smiles_col
            ].tolist()
            raise ValueError(
                f"Duplicate SMILES in annotations CSV: {duplicates[:5]}"
            )

        extra_annotations = (
            df.set_index(smiles_col).to_dict(orient="index")
        )

    emb_array = []
    annotations = []
    entities = []
    predictions = []
    targets = []

    for smi, emb_dict in embeddings.items():

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            print(f"Warning: Could not parse SMILES '{smi}'. Skipping.")
            continue

        annotation = motif_annotation(
            mol=mol,
            motif_dict=motif_dict,
            level="graph"
        )

        descriptors_mol = calculate_descriptors(mol)
        annotation.update(descriptors_mol)

        # Add optional annotations matched by SMILES
        if annotations_csv is not None:
            if smi not in extra_annotations:
                raise ValueError(
                    f"SMILES '{smi}' is missing from {annotations_csv}."
                )

            for key, value in extra_annotations[smi].items():
                if key in annotation:
                    raise ValueError(
                        f"Additional annotation '{key}' conflicts "
                        "with an existing annotation."
                    )

                annotation[key] = value

        graph_embed = emb_dict[embedding_key]

        emb_array.append(
            graph_embed.cpu().numpy().astype(np.float16)
        )

        annotations.append(annotation)
        entities.append(smi)
        predictions.append(emb_dict["prediction"])
        targets.append(emb_dict["target"])

    annotations, retained_annotations = filter_constant_annotations(
        annotations
    )

    return {
        "embeddings": np.stack(emb_array, axis=0),
        "annotations": annotations,
        "entities": entities,
        "prediction": np.array(predictions),
        "target": np.array(targets),
        "molecular_descriptors": [
            name for name in descriptors_mol
            if name in retained_annotations
        ],
        "retained_annotations": retained_annotations,
    }




