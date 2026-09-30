from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Subset
import numpy as np

def get_scaffold(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        # sanitize again
        Chem.SanitizeMol(mol)

        # remove problematic stereo information
        Chem.RemoveStereochemistry(mol)

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol
        )

        return scaffold

    except Exception as e:
        print(f"Failed scaffold for SMILES: {smiles}")
        print(e)
        return None


def scaffold_split(
    dataset,
    splits=(0.8, 0.1, 0.1),
    seed=42
):
    train_pct, val_pct, test_pct = splits

    if not np.isclose(sum(splits), 1.0):
        raise ValueError(
            f"Splits must sum to 1. Got {splits}"
        )

    rng = np.random.default_rng(seed)

    entities = np.asarray(dataset.entities)
    unique_smiles = np.unique(entities)

    print("Total embeddings:", len(entities))
    print("Unique molecules:", len(unique_smiles))

    # ---------------------------------------------------------
    # Group molecules by scaffold
    # ---------------------------------------------------------
    scaffold_groups = {}

    for smi in unique_smiles:

        scaffold = get_scaffold(smi)

        if scaffold is None:
            scaffold = f"INVALID_{smi}"

        scaffold_groups.setdefault(scaffold, []).append(smi)

    # ---------------------------------------------------------
    # Target number of molecules
    # ---------------------------------------------------------
    total_molecules = len(unique_smiles)

    train_size = train_pct * total_molecules
    val_size = val_pct * total_molecules

    # ---------------------------------------------------------
    # Shuffle scaffolds with seed
    # ---------------------------------------------------------
    scaffolds = list(scaffold_groups.keys())
    rng.shuffle(scaffolds)

    # Then sort by scaffold size, keeping the random order
    # for scaffolds with equal size.
    scaffolds = sorted(
        scaffolds,
        key=lambda scaffold: len(scaffold_groups[scaffold]),
        reverse=True
    )

    # ---------------------------------------------------------
    # Assign scaffolds GROVER-style
    # ---------------------------------------------------------
    train_scaffolds = []
    val_scaffolds = []
    test_scaffolds = []

    train_smiles = []
    val_smiles = []
    test_smiles = []

    for scaffold in scaffolds:

        molecules = scaffold_groups[scaffold]
        scaffold_size = len(molecules)

        if len(train_smiles) + scaffold_size <= train_size:

            train_smiles.extend(molecules)
            train_scaffolds.append(scaffold)

        elif len(val_smiles) + scaffold_size <= val_size:

            val_smiles.extend(molecules)
            val_scaffolds.append(scaffold)

        else:

            test_smiles.extend(molecules)
            test_scaffolds.append(scaffold)

    # Convert to sets
    train_smiles = set(train_smiles)
    val_smiles = set(val_smiles)
    test_smiles = set(test_smiles)

    # ---------------------------------------------------------
    # Map molecules to embedding rows
    # ---------------------------------------------------------
    train_idx = np.where(
        np.isin(entities, list(train_smiles))
    )[0]

    val_idx = np.where(
        np.isin(entities, list(val_smiles))
    )[0]

    test_idx = np.where(
        np.isin(entities, list(test_smiles))
    )[0]

    # ---------------------------------------------------------
    # Checks
    # ---------------------------------------------------------
    assert train_smiles.isdisjoint(val_smiles)
    assert train_smiles.isdisjoint(test_smiles)
    assert val_smiles.isdisjoint(test_smiles)

    train_scaf = set(train_scaffolds)
    val_scaf = set(val_scaffolds)
    test_scaf = set(test_scaffolds)

    assert train_scaf.isdisjoint(val_scaf)
    assert train_scaf.isdisjoint(test_scaf)
    assert val_scaf.isdisjoint(test_scaf)

    # ---------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------
    print("\nMolecule counts:")
    print(f"Total: {len(unique_smiles)}")
    print(
        f"Train: {len(train_smiles)} "
        f"({len(train_smiles) / len(unique_smiles):.2%})"
    )
    print(
        f"Val:   {len(val_smiles)} "
        f"({len(val_smiles) / len(unique_smiles):.2%})"
    )
    print(
        f"Test:  {len(test_smiles)} "
        f"({len(test_smiles) / len(unique_smiles):.2%})"
    )

    print("\nEmbedding counts:")
    print(f"Total: {len(entities)}")
    print(
        f"Train: {len(train_idx)} "
        f"({len(train_idx) / len(entities):.2%})"
    )
    print(
        f"Val:   {len(val_idx)} "
        f"({len(val_idx) / len(entities):.2%})"
    )
    print(
        f"Test:  {len(test_idx)} "
        f"({len(test_idx) / len(entities):.2%})"
    )

    print("\nScaffold counts:")
    print("Train:", len(train_scaffolds))
    print("Val:", len(val_scaffolds))
    print("Test:", len(test_scaffolds))

    print("\nScaffold overlap:")
    print("Train ∩ Val:", train_scaf & val_scaf)
    print("Train ∩ Test:", train_scaf & test_scaf)
    print("Val ∩ Test:", val_scaf & test_scaf)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
