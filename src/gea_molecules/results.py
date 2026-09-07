import torch
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
from torch.utils.data import DataLoader, Subset
import numpy as np
import colorsys


def split_solubility_loaders(
    test_loader,
    high_threshold=-1.0,
    low_threshold=-6.0,
):
    dataset = test_loader.dataset

    high_indices = []
    low_indices = []

    # Store target for each molecule
    molecule_targets = {}

    for idx in range(len(dataset)):

        sample = dataset[idx]

        smiles = sample["entity"]
        target = sample["target"]

        # Convert target to a Python float
        if torch.is_tensor(target):
            target = target.item()
        else:
            target = float(target)

        # Check consistency if molecule appears multiple times
        if smiles in molecule_targets:
            previous_target = molecule_targets[smiles]

            if not np.isclose(target, previous_target):
                raise ValueError(
                    f"Molecule {smiles} has inconsistent targets: "
                    f"{previous_target} vs {target}"
                )
        else:
            molecule_targets[smiles] = target

    # Now assign EVERY atom belonging to a molecule
    # to the same split
    for idx in range(len(dataset)):

        smiles = dataset[idx]["entity"]
        target = molecule_targets[smiles]

        if target > high_threshold:
            high_indices.append(idx)

        elif target < low_threshold:
            low_indices.append(idx)

    high_dataset = Subset(dataset, high_indices)
    low_dataset = Subset(dataset, low_indices)

    # Create loaders
    high_loader = DataLoader(
        high_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
    )

    low_loader = DataLoader(
        low_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
    )

    # Diagnostics
    high_smiles = {
        dataset[idx]["entity"]
        for idx in high_indices
    }

    low_smiles = {
        dataset[idx]["entity"]
        for idx in low_indices
    }

    print("Solubility split")
    print("----------------")
    print(f"High solubility molecules (logS > {high_threshold}): "
          f"{len(high_smiles)}")
    print(f"Very low solubility molecules (logS < {low_threshold}): "
          f"{len(low_smiles)}")

    print("\nAtom counts")
    print(f"High solubility atoms: {len(high_indices)}")
    print(f"Very low solubility atoms: {len(low_indices)}")

    print("\nMolecule overlap:")
    print("High ∩ Low:", high_smiles & low_smiles)

    return high_loader, low_loader

def extract_activations(
        sae_model, 
        data_loader,
        max_features,
        selected_features,
        device = "cuda"
):
    sae_model.to(device)
    sae_model.eval()
    with torch.no_grad():
        for batch in data_loader:

            embeddings = batch["embedding"].to(device)
            smiles = batch["entity"]
            max_features = max_features.to(device)

            z_sae, _ = sae_model(embeddings)

            normalized_sae_activations = z_sae / max_features
            feature_act = normalized_sae_activations[:, selected_features]

    



def selected_feature_activation(
    sae_model,
    data_loader,
    max_features,
    feature,
    activation_threshold=0.1,
    device="cuda",
):
    sae_model.to(device)
    sae_model.eval()

    feature_activation = {}

    with torch.no_grad():
        for batch in data_loader:

            embeddings = batch["embedding"].to(device)
            smiles = batch["entity"]
            max_features = max_features.to(device)

            z_sae, _ = sae_model(embeddings)

            normalized_sae_activations = z_sae / max_features
            feature_act = normalized_sae_activations[:, feature]

            # Collect ALL atom activations for each molecule
            for smi, act in zip(smiles, feature_act):
                act = act.item()
                if smi not in feature_activation:
                    feature_activation[smi] = []

                feature_activation[smi].append(act)

    # Keep the COMPLETE activation list only for molecules
    # where at least one atom exceeds the threshold
    feature_activation = {
        smi: activations
        for smi, activations in feature_activation.items()
        if max(activations) > activation_threshold
    }

    return feature_activation

def all_features_activation(test_results, sae_model, data_loader, 
                            max_features, activation_threshold = 0.1, device = 'cuda'):

    features_activations = {}
    for concept, info_list in test_results.items():
        for info in info_list:
            feature = info['feature']
            smiles_dict = selected_feature_activation(
                sae_model = sae_model, 
                data_loader = data_loader, 
                max_features = max_features, 
                feature = feature, 
                activation_threshold=activation_threshold,
                device = device
            )

            features_activations[(concept, feature)] = smiles_dict

    return features_activations

def draw_activation(mol, activations, threshold=0.0):

    # Convert tensors to floats
    activations = [
        x.detach().cpu().item()
        if hasattr(x, "detach")
        else float(x)
        for x in activations
    ]

    if len(activations) != mol.GetNumAtoms():
        raise ValueError(
            f"Number of activations ({len(activations)}) "
            f"does not match number of atoms ({mol.GetNumAtoms()})"
        )

    highlight_atoms = []
    atom_colors = {}

    # Nicer blue -> red colors
    blue = (0.392, 0.584, 0.929)
    red = (0.980, 0.502, 0.447)

    for i, value in enumerate(activations):

        if value > threshold:

            highlight_atoms.append(i)

            # Clamp activation to [0, 1]
            value = max(0.0, min(1.0, value))

            # Blue -> red interpolation
            atom_colors[i] = tuple(
                blue[j] * (1 - value) + red[j] * value
                for j in range(3)
            )

    # RDKit drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)

    options = drawer.drawOptions()
    options.useBWAtomPalette()


    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=[],          # <-- do NOT highlight bonds
    )

    drawer.FinishDrawing()

    return Image.open(
        io.BytesIO(drawer.GetDrawingText())
    )