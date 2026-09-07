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


def scaffold_split(dataset, splits, seed=42):

    train_pct, val_pct, test_pct = splits

    if not np.isclose(sum(splits), 1.0):
        raise ValueError(
            f"Splits must sum to 1. Got {splits}"
        )

    rng = np.random.default_rng(seed)

    # Entity associated with each embedding
    entities = np.asarray(dataset.entities)

    # Unique molecules
    unique_entities = np.unique(entities)

    # Group molecules by scaffold
    scaffold_groups = {}

    for smi in unique_entities:

        scaffold = get_scaffold(smi)

        if scaffold is None:
            scaffold = f"INVALID_{smi}"

        scaffold_groups.setdefault(scaffold, []).append(smi)

    # Shuffle scaffolds
    scaffolds = list(scaffold_groups.keys())
    rng.shuffle(scaffolds)

    # Split scaffolds
    n_scaffolds = len(scaffolds)

    train_end = int(train_pct * n_scaffolds)
    val_end = int((train_pct + val_pct) * n_scaffolds)

    train_scaffolds = scaffolds[:train_end]
    val_scaffolds = scaffolds[train_end:val_end]
    test_scaffolds = scaffolds[val_end:]

    # Convert scaffold assignments to molecule assignments
    train_entities = {
        smi
        for scaffold in train_scaffolds
        for smi in scaffold_groups[scaffold]
    }

    val_entities = {
        smi
        for scaffold in val_scaffolds
        for smi in scaffold_groups[scaffold]
    }

    test_entities = {
        smi
        for scaffold in test_scaffolds
        for smi in scaffold_groups[scaffold]
    }

    # Map molecule assignments back to embedding rows
    train_idx = np.where(
        np.isin(entities, list(train_entities))
    )[0]

    val_idx = np.where(
        np.isin(entities, list(val_entities))
    )[0]

    test_idx = np.where(
        np.isin(entities, list(test_entities))
    )[0]

    # Checks
    assert len(set(train_entities) & set(val_entities)) == 0
    assert len(set(train_entities) & set(test_entities)) == 0
    assert len(set(val_entities) & set(test_entities)) == 0

    print("\nCounts:")
    print("Total embeddings:", len(entities))
    print("Train embeddings:", len(train_idx))
    print("Val embeddings:", len(val_idx))
    print("Test embeddings:", len(test_idx))

    print("\nUnique molecules:")
    print("Train:", len(train_entities))
    print("Val:", len(val_entities))
    print("Test:", len(test_entities))

    print("\nUnique scaffolds:")
    print("Train:", len(train_scaffolds))
    print("Val:", len(val_scaffolds))
    print("Test:", len(test_scaffolds))

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def scaffold_split(dataset, splits=(0.8, 0.1, 0.1), seed=42):

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

    scaffold_groups = {}

    for smi in unique_smiles:

        scaffold = get_scaffold(smi)

        # Keep invalid molecules separate
        if scaffold is None:
            scaffold = f"INVALID_{smi}"

        scaffold_groups.setdefault(scaffold, []).append(smi)

    scaffolds = list(scaffold_groups.keys())
    rng.shuffle(scaffolds)

 
    # assign scaffolds based on the number of unique molecules they contain.
    total_molecules = len(unique_smiles)

    target_train = train_pct * total_molecules
    target_val = val_pct * total_molecules
    target_test = test_pct * total_molecules

    train_scaffolds = []
    val_scaffolds = []
    test_scaffolds = []

    train_count = 0
    val_count = 0
    test_count = 0

    # Sort scaffolds by size, largest first.
    # This makes the greedy assignment more stable.
    scaffolds = sorted(
        scaffolds,
        key=lambda s: len(scaffold_groups[s]),
        reverse=True
    )

    # Assign each scaffold to the split that is currently
    # furthest below its target.
    for scaffold in scaffolds:

        scaffold_size = len(scaffold_groups[scaffold])

        train_fraction = train_count / target_train if target_train > 0 else np.inf
        val_fraction = val_count / target_val if target_val > 0 else np.inf
        test_fraction = test_count / target_test if target_test > 0 else np.inf

        fractions = {
            "train": train_fraction,
            "val": val_fraction,
            "test": test_fraction
        }

        split = min(fractions, key=fractions.get)

        if split == "train":
            train_scaffolds.append(scaffold)
            train_count += scaffold_size

        elif split == "val":
            val_scaffolds.append(scaffold)
            val_count += scaffold_size

        else:
            test_scaffolds.append(scaffold)
            test_count += scaffold_size

    train_smiles = {
        smi
        for scaffold in train_scaffolds
        for smi in scaffold_groups[scaffold]
    }

    val_smiles = {
        smi
        for scaffold in val_scaffolds
        for smi in scaffold_groups[scaffold]
    }

    test_smiles = {
        smi
        for scaffold in test_scaffolds
        for smi in scaffold_groups[scaffold]
    }

    train_idx = np.where(
        np.isin(entities, list(train_smiles))
    )[0]

    val_idx = np.where(
        np.isin(entities, list(val_smiles))
    )[0]

    test_idx = np.where(
        np.isin(entities, list(test_smiles))
    )[0]


    assert len(train_smiles & val_smiles) == 0
    assert len(train_smiles & test_smiles) == 0
    assert len(val_smiles & test_smiles) == 0


    train_scaf = {
        get_scaffold(smi)
        for smi in train_smiles
    }

    val_scaf = {
        get_scaffold(smi)
        for smi in val_smiles
    }

    test_scaf = {
        get_scaffold(smi)
        for smi in test_smiles
    }

    # Remove None if any invalid SMILES were encountered
    train_scaf.discard(None)
    val_scaf.discard(None)
    test_scaf.discard(None)

    assert len(train_scaf & val_scaf) == 0
    assert len(train_scaf & test_scaf) == 0
    assert len(val_scaf & test_scaf) == 0


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
