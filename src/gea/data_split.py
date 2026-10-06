import torch
import numpy as np

def random_split_dataset(dataset, splits, seed, level):
    """Randomly split a dataset into train, validation and test sets.

    For graph-level data, individual samples are randomly split.

    For node-level data, samples belonging to the same entity/graph are
    kept in the same split.
    """
    train_frac, val_frac, test_frac = splits

    if not np.isclose(sum(splits), 1.0):
        raise ValueError("Split fractions must sum to 1.")

    rng = np.random.default_rng(seed)

    if level == "graph":
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

    elif level == "node":
        entities = np.asarray(dataset.entities)

        if len(entities) != len(dataset):
            raise ValueError(
                "Number of entities must match dataset length."
            )

        unique_entities = np.unique(entities)
        rng.shuffle(unique_entities)

        n_entities = len(unique_entities)

        n_train = int(n_entities * train_frac)
        n_val = int(n_entities * val_frac)

        train_entities = set(unique_entities[:n_train])
        val_entities = set(
            unique_entities[n_train:n_train + n_val]
        )
        test_entities = set(
            unique_entities[n_train + n_val:]
        )

        train_indices = np.array([
            i for i, entity in enumerate(entities)
            if entity in train_entities
        ])

        val_indices = np.array([
            i for i, entity in enumerate(entities)
            if entity in val_entities
        ])

        test_indices = np.array([
            i for i, entity in enumerate(entities)
            if entity in test_entities
        ])

        return (
            torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, val_indices),
            torch.utils.data.Subset(dataset, test_indices),
        )

    else:
        raise ValueError(
            f"Unknown level '{level}'. Use 'graph' or 'node'."
        )

    n = len(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
        torch.utils.data.Subset(dataset, test_indices),
    )