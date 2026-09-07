import torch
from rdkit import Chem
import numpy as np
from rdkit import Chem
from rdkit.Chem import Fragments, Descriptors, Crippen, Lipinski, rdMolDescriptors
import inspect

def bondConceptAnnotation(mol):
    n_atoms = mol.GetNumAtoms()
    pi_mask = [0] * n_atoms
    pi_double_mask = [0] * n_atoms
    pi_triple_mask = [0] * n_atoms
    pi_aromatic_mask = [0] * n_atoms

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Double bond
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            pi_mask[i] = 1
            pi_mask[j] = 1

            pi_double_mask[i] = 1
            pi_double_mask[j] = 1

        # Triple bond
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            pi_mask[i] = 1
            pi_mask[j] = 1

            pi_triple_mask[i] = 1
            pi_triple_mask[j] = 1

        # Aromatic bond
        elif bond.GetIsAromatic():
            pi_mask[i] = 1
            pi_mask[j] = 1

            pi_aromatic_mask[i] = 1
            pi_aromatic_mask[j] = 1

    return {
        "pi bond": pi_mask,
        "pi bond - double": pi_double_mask,
        "pi bond - triple": pi_triple_mask,
        "pi bond - aromatic": pi_aromatic_mask,
    }

def motifAnnotation(mol, motif_dict):
    mol_dict = {}

    # -----------------------------------------
    # SMARTS-based concepts
    # -----------------------------------------

    for motif, smarts in motif_dict.items():
        patt = Chem.MolFromSmarts(smarts)
        mask = [0]*mol.GetNumAtoms()
        for match in mol.GetSubstructMatches(patt):
            for atom_idx in match:
                mask[atom_idx] = 1
    
        mol_dict[motif] = mask

    # -----------------------------------------
    # Bond-based concepts
    # -----------------------------------------

    bond_annotations = bondConceptAnnotation(mol)

    mol_dict.update(bond_annotations)

    return mol_dict

def fragments_count(mol):

    mol_fragments={}

    halogen_patterns = {
        "F": Chem.MolFromSmarts("[F]"),
        "Cl": Chem.MolFromSmarts("[Cl]"),
        "Br": Chem.MolFromSmarts("[Br]"),
        "I": Chem.MolFromSmarts("[I]"),
    }

    for name, pattern in halogen_patterns.items():
        mol_fragments[name] = len(mol.GetSubstructMatches(pattern))
    
    for name in dir(Fragments):
        if name.startswith("fr_"):
            func = getattr(Fragments, name)
            if callable(func):
                count = func(mol)
                mol_fragments[name] = count

    return mol_fragments


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

def mol_annotation(mol):

    frag_counts = fragments_count(mol)

    total_annotation = {}

    for key, value in frag_counts.items():
        new_keys = [key + count for count in ["_1", "_2", "_3+"]]
        for new_key in new_keys:
            if new_key not in total_annotation:
                total_annotation[new_key] = 0
        if value == 1:
            total_annotation[key + "_1"] = 1
        elif value == 2:
            total_annotation[key + "_2"] = 1
        elif value >= 3:
            total_annotation[key + "_3+"] = 1

    descriptors_mol = calculate_descriptors(mol)

    for key, value in descriptors_mol.items():
        total_annotation[key] = value

    return total_annotation

def annotate_node_GroverEmbeds(embeddings_path, motif_dict, embedding_key="atom_from_atom"):

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

        node_embeds = emb_dict[embedding_key] # We need to see if we want to consider atom_from_bond

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

def annotate_graph_GroverEmbeds(embeddings_path, embedding_key = "graph_from_atom_from_atom"):

    embeddings = torch.load(embeddings_path)

    emb_array = []
    annotations = []
    entities = []
    predictions = []
    targets = []

    for smi, emb_dict in embeddings.items():

        mol = Chem.MolFromSmiles(smi)

        annotation = mol_annotation(mol)

        graph_embed = emb_dict[embedding_key]

        emb_array.append(
            graph_embed.cpu().numpy().astype(np.float16)
        )

        annotations.append(annotation)
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




