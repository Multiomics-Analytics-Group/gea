import argparse
import re
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Subset
from gea.dataloader import EmbeddingDataset
from gea.gea import ShallowSAE
from gea.analysis import (
    compute_feature_maxima,
    detect_numeric_annotation_types,
)
from pathlib import Path
import json
from scipy.stats import spearmanr


def get_embeddings(batch):
    """
    Return the embeddings passed to the SAE.

    Both graph and node embeddings are expected to be stored under
    batch['embedding']. The difference is how samples are grouped later.
    """
    embeddings = batch["embedding"]

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected embeddings with shape (n_samples, embedding_dim), "
            f"got {tuple(embeddings.shape)}."
        )

    return embeddings


def calculate_annotation_prevalence(
    dataset,
    evaluation_indices,
    class_names=None,
    classification_threshold=0.5,
):
    """
    Calculate annotation counts and prevalence across correctly and
    incorrectly classified samples for each target class.

    Only binary annotations are considered. Continuous and count annotations
    are excluded. Class-specific annotations (e.g. annotations ending in
    '_soluble' or '_insoluble') are also excluded.

    Parameters
    ----------
    dataset : EmbeddingDataset
        Dataset containing molecular annotations, target labels and model
        predictions. Annotations are expected to be already processed and
        binary (0/1).

    evaluation_indices : list or array-like
        Indices of the samples to include in the analysis.

    class_names : dict, optional
        Mapping between integer class labels and their corresponding names.
        Defaults to {0: 'class_0', 1: 'class_1'}.

    classification_threshold : float, optional
        Threshold used to convert prediction probabilities into binary
        class labels. Predictions greater than or equal to this threshold
        are assigned to class 1. Defaults to 0.5.

    Returns
    -------
    dict
        Dictionary containing:

        annotation_counts : pd.DataFrame
            Number of samples containing each annotation, with columns
            corresponding to correctly and incorrectly classified samples
            in each target class.

        annotation_prevalence : pd.DataFrame
            Proportion of samples containing each annotation within each
            classification group. Calculated as annotation_counts divided
            by the corresponding group size.

        class_prediction_group_sizes : pd.Series
            Number of samples in each classification group.

        binary_variables : list
            Names of the binary annotations included in the analysis.

    Notes
    -----
    Samples are grouped according to their actual target class and whether
    their predicted class matches the target.

    Annotation prevalence represents the proportion of samples containing
    a given annotation, rather than the frequency of its occurrences within
    individual samples.
    """

    if class_names is None:
        class_names = {
            0: "class_0",
            1: "class_1",
        }

    # Identify binary annotations.
    continuous_variables, count_variables, binary_variables = detect_numeric_annotation_types(dataset)
    numeric_variables = continuous_variables + count_variables

    # Extract annotations for the selected samples.
    annotations_df = pd.DataFrame.from_records(
        [dataset.annotations[int(i)] for i in evaluation_indices]
    ).fillna(0)

    # Extract targets and predictions.
    targets = np.asarray([
        dataset[int(i)]["target"].item()
        if torch.is_tensor(dataset[int(i)]["target"])
        else dataset[int(i)]["target"]
        for i in evaluation_indices
    ]).astype(int)

    predictions = np.asarray([
        dataset[int(i)]["prediction"].item()
        if torch.is_tensor(dataset[int(i)]["prediction"])
        else dataset[int(i)]["prediction"]
        for i in evaluation_indices
    ])

    # Keep only binary annotations present in the dataset.
    binary_variables = [
        variable for variable in binary_variables
        if variable in annotations_df.columns
    ]

    annotations_df = annotations_df[binary_variables].copy()

    # Ensure binary numeric values.
    annotations_df = annotations_df.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    annotations_df = (annotations_df > 0).astype(int)

    # Remove class-specific annotations.
    suffixes = tuple(
        f"_{name}" for name in class_names.values()
    )

    annotations_df = annotations_df.loc[
        :,
        [
            col for col in annotations_df.columns
            if not str(col).endswith(suffixes)
        ]
    ]

    # Convert prediction probabilities into class labels.
    predictions = (
        predictions >= classification_threshold
    ).astype(int)

    groups = {}

    # Define correctly and incorrectly classified groups.
    for class_id, class_name in class_names.items():

        groups[f"correct_{class_name}"] = (
            (targets == class_id) & (predictions == class_id)
        )

        groups[f"misclassified_{class_name}"] = (
            (targets == class_id) & (predictions != class_id)
        )

    # Calculate group sizes.
    class_prediction_group_sizes = pd.Series({
        name: int(mask.sum())
        for name, mask in groups.items()
    })

    # Calculate annotation counts.
    annotation_counts = pd.DataFrame({
        name: annotations_df.loc[mask].sum(axis=0)
        for name, mask in groups.items()
    })

    # Calculate annotation prevalence.
    annotation_prevalence = annotation_counts.div(
        class_prediction_group_sizes.replace(0, np.nan),
        axis=1,
    )

    print("\nAnnotation group sizes:")
    print(class_prediction_group_sizes)

    print("\nNumber of concepts:", len(annotation_counts))

    return {
        "annotation_counts": annotation_counts,
        "annotation_prevalence": annotation_prevalence,
        "class_prediction_group_sizes": class_prediction_group_sizes,
        "binary_variables": binary_variables,
        "numeric_variables": numeric_variables
    }

def calculate_numeric_correlations_by_group(
    dataset,
    evaluation_indices,
    sample_results,
    numeric_variables,
    class_names=None,
    classification_threshold=0.5,
):
    """
    Calculate Spearman correlations between SAE feature activations
    and numeric annotations independently in each classification group.

    Returns
    -------
    dict
        {
            feature_id: {
                annotation_name: {
                    group_name: {
                        "correlation": float,
                        "n": int,
                    }
                }
            }
        }
    """

    if class_names is None:
        class_names = {
            0: "class_0",
            1: "class_1",
        }

    # Recover activations, targets and predictions.
    results = list(sample_results.values())

    activations = np.stack([
        result["activations"].numpy()
        for result in results
    ])

    targets = np.asarray([
        result["target"] for result in results
    ]).astype(int)

    predictions = np.asarray([
        result["prediction"] for result in results
    ])

    predictions = (
        predictions >= classification_threshold
    ).astype(int)

    # Define classification groups.
    groups = {}

    for class_id, class_name in class_names.items():

        groups[f"correct_{class_name}"] = (
            (targets == class_id) & (predictions == class_id)
        )

        groups[f"misclassified_{class_name}"] = (
            (targets == class_id) & (predictions != class_id)
        )

    # Recover numeric annotations.
    annotation_values = {}

    for name in numeric_variables:

        values = []

        for idx in evaluation_indices:

            value = dataset.annotations[int(idx)].get(name, np.nan)

            if torch.is_tensor(value):
                value = value.item()

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan

            values.append(value)

        annotation_values[name] = np.asarray(values, dtype=float)

    # Calculate correlations for each feature, annotation and group.
    numeric_correlations = {}

    for feature in range(activations.shape[1]):

        numeric_correlations[feature] = {}

        feature_values = activations[:, feature]

        for name, values in annotation_values.items():

            numeric_correlations[feature][name] = {}

            for group_name, mask in groups.items():

                valid = (
                    mask
                    & np.isfinite(values)
                    & np.isfinite(feature_values)
                )

                n = int(valid.sum())
                correlation = np.nan

                if (
                    n >= 2
                    and np.unique(values[valid]).size >= 2
                    and np.unique(feature_values[valid]).size >= 2
                ):
                    correlation = spearmanr(
                        values[valid],
                        feature_values[valid],
                    ).statistic

                numeric_correlations[feature][name][group_name] = {
                    "correlation": correlation,
                    "n": n,
                }

    return numeric_correlations

def main(args):
    
    device = args.device if (
        args.device.startswith("cpu") or torch.cuda.is_available()
    ) else "cpu"

    print("Loading dataset...")
    dataset = EmbeddingDataset(args.embeddings_path)

    splits = torch.load(args.splits_path, 
                        map_location="cpu",
                        weights_only=False)

    if args.split not in splits:
        raise ValueError(
            f"Split '{args.split}' not found. "
            f"Available splits: {list(splits)}"
        )

    train_indices = splits["train"]
    evaluation_indices = splits[args.split]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    evaluation_loader = DataLoader(
        Subset(dataset, evaluation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialise the trained SAE.
    sae = ShallowSAE(
        in_dim=args.d_z,
        latent_dim=args.d_z * args.latent_multiplier,
        sparsity_weight=args.sparsity_weight,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    sae.load_state_dict(checkpoint)
    sae.eval()

    # Always compute maxima from training data, regardless of selected split.
    print("Computing feature maxima on training data...")
    maxima_path = Path(args.maxima_path)

    if maxima_path.exists():
        maxima = torch.load(
            maxima_path,
            map_location="cpu",
            weights_only=False,
        )

        train_feature_maxima = maxima["max_features"]

        print(f"Loaded training feature maxima from {maxima_path}")

    else:
        train_feature_maxima, _, _ = (
            compute_feature_maxima(
                sae_model=sae,
                data_loader=train_loader,
                device=device,
            )
        )

        maxima_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "max_features": train_feature_maxima,
            },
            maxima_path,
        )

        print(f"Saved training feature maxima to {maxima_path}")

    train_feature_maxima = torch.as_tensor(
        train_feature_maxima, device=device
    ).reshape(-1)

    print(f"Processing split: {args.split}")

    sample_results = {}
    with torch.inference_mode():
        for batch in evaluation_loader:
            embeddings = get_embeddings(
                batch,
            ).to(device)

            feature_activations, _ = sae(embeddings)

            targets = torch.as_tensor(batch["target"]).reshape(-1).cpu()
            predictions = torch.as_tensor(batch["prediction"]).reshape(-1).cpu()
            entities = list(batch["entity"])

            n_samples = embeddings.shape[0]

            if not (
                len(targets) == len(predictions) == len(entities) == n_samples
            ):
                raise ValueError(
                    "Targets, predictions and entities must contain "
                    "one value per embedding."
                )

            for i in range(n_samples):
                sample_id = f"item_{len(sample_results)}"
                sample_results[sample_id] = {
                    "entity": entities[i],
                    "target": targets[i].item(),
                    "prediction": predictions[i].item(),
                    "activations": feature_activations[i].detach().cpu(),
                }

    print("Calculating annotation prevalence...")

    class_names = (
        {int(k): v for k, v in args.class_names.items()}
        if args.class_names is not None
        else {0: "class_0", 1: "class_1"}
    )

    annotation_statistics = calculate_annotation_prevalence(
        dataset=dataset,
        evaluation_indices=evaluation_indices,
        class_names=class_names,
        classification_threshold=args.classification_threshold,
    )

    numeric_variables = annotation_statistics["numeric_variables"]

    numeric_correlations_by_group = {}

    if numeric_variables:

        print("Calculating numeric annotation correlations by group...")

        numeric_correlations_by_group = (
            calculate_numeric_correlations_by_group(
                dataset=dataset,
                evaluation_indices=evaluation_indices,
                sample_results=sample_results,
                numeric_variables=numeric_variables,
                class_names=class_names,
                classification_threshold=args.classification_threshold,
            )
        )

    output = {
        "split": args.split,
        "train_feature_maxima": train_feature_maxima.cpu(),
        "results": sample_results,

        # Annotation statistics.
        "annotation_counts": annotation_statistics["annotation_counts"],
        "annotation_prevalence": annotation_statistics["annotation_prevalence"],
        "class_prediction_group_sizes": annotation_statistics["class_prediction_group_sizes"],

        # Annotation variable types.
        "binary_variables": annotation_statistics["binary_variables"],
        "numeric_variables": numeric_variables,

        # Numeric annotation correlations.
        "numeric_correlations_by_group": numeric_correlations_by_group,

        # Classification configuration.
        "class_names": class_names,
        "classification_threshold": args.classification_threshold,
    }

    print("Saving results to:", args.sae_annotation_association_path)
    torch.save(output, args.sae_annotation_association_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--sae_annotation_association_path", default="sae_annotation_association_results.pt")

    parser.add_argument(
        "--maxima_path",
        type=Path,
        required=True,
        help="Path to the saved training feature maxima.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )

    parser.add_argument("--d_z", type=int, default=1200)
    parser.add_argument("--latent_multiplier", type=int, default=8)
    parser.add_argument("--sparsity_weight", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--classification_threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--class_names",
        type=json.loads,
        default=None,
        help='Dictionary mapping class labels to names (e.g. \'{"0": "insoluble", "1": "soluble"}\').',
    )


    main(parser.parse_args())
