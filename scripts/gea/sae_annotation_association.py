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
    incorrectly classified soluble and insoluble molecules.

    Binary variables are counted directly.
    Count variables are converted to binary presence/absence.
    Continuous variables are excluded.
    """

    # Identify annotation types from the original annotations.
    _, count_variables, binary_variables = (
        detect_numeric_annotation_types(dataset)
    )

    selected_variables = list(dict.fromkeys(
        binary_variables + count_variables
    ))

    # Extract annotations for the selected molecules only.
    annotations_df = pd.DataFrame.from_records(
        [dataset.annotations[int(i)] for i in evaluation_indices]
    ).fillna(0)

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

    # Keep only binary and count variables.
    selected_variables = [
        variable for variable in selected_variables
        if variable in annotations_df.columns
    ]

    annotations_df = annotations_df[selected_variables].copy()

    # Ensure numeric values.
    annotations_df = annotations_df.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    # Count variables represent presence/absence, not number of occurrences.
    annotations_df[count_variables] = (
        annotations_df[
            [v for v in count_variables if v in annotations_df.columns]
        ] > 0
    ).astype(int)

    # Binary annotations are also represented as 0/1.
    annotations_df[binary_variables] = (
        annotations_df[
            [v for v in binary_variables if v in annotations_df.columns]
        ] > 0
    ).astype(int)

    # Remove class-specific suffixes, if present.
    suffixes = tuple(
        f"_{name}" for name in class_names.values()
    )

    annotations_df.columns = [
        re.sub(
            rf"(?:{'|'.join(map(re.escape, suffixes))})$",
            "",
            str(col),
        )
        for col in annotations_df.columns
    ]

    # Merge columns with the same concept name, if any.
    annotations_df = (
        annotations_df.T.groupby(level=0).max().T
    )

    # Determine predicted labels.
    targets = np.asarray(targets).astype(int)
    predictions = (
        np.asarray(predictions) >= classification_threshold
    ).astype(int)

    if class_names is None:
        class_names = {
            0: "class_0",
            1: "class_1",
        }

    groups = {}



    # Define the four groups.
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
        "count_variables": count_variables,
    }


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
        "count_variables": annotation_statistics["count_variables"],

        # Classification configuration.
        "class_names": class_names,
        "classification_threshold": args.classification_threshold,
    }

    print("Saving results to:", args.output_path)
    torch.save(output, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", default="class_activations.pt")

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
