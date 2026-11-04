from gea.dataloader import EmbeddingDataset
from gea.analysis import (
    gea_annotation, concept_feature_test, select_concept_features, 
    calculate_quartile_thresholds, binarize_all_annotations, calculate_count_categories,
    detect_numeric_annotation_types, select_numeric_features, compute_feature_maxima,
    add_class_specific_annotations
)
from gea.gea import ShallowSAE
import torch
from torch.utils.data import DataLoader, Subset
import argparse
import json

def main(args):

    print("Loading embeddings from:", args.embeddings_path, "Initializing dataset...")

    numeric_variables = []
    numeric_correlations = {}
    numeric_feature_pairs = {}

    continuous_variables = []
    count_variables = []
    binary_variables = []

    emb_data = EmbeddingDataset(args.embeddings_path)

    splits = torch.load(args.splits_path, weights_only=False)
    
    train_indices = splits["train"]

    args.class_names = {
        int(label): name
        for label, name in args.class_names.items()
    }

    if args.is_numeric:

        print("Preparing numeric annotations...")

        continuous_variables, count_variables, binary_variables = (
            detect_numeric_annotation_types(emb_data)
        )

        numeric_variables = continuous_variables + count_variables

        quartile_thresholds = calculate_quartile_thresholds(
            emb_data,
            train_indices,
            continuous_variables,
        )

        count_categories = calculate_count_categories(
            emb_data,
            train_indices,
            count_variables,
            min_category_count=args.min_category_count
        )

        for idx in range(len(emb_data)):

            emb_data.annotations[idx] = binarize_all_annotations(
                emb_data,
                idx,
                quartile_thresholds,
                count_categories,
                binary_variables,
            )

    else:
        print("Using binary annotations...")
        binary_variables = list(emb_data.annotations[0].keys())

    print("Adding class-specific annotations...")
    targets = emb_data.targets

    # Detect types again using the binarised annotations
    continuous_variables, count_variables, binary_variables = (
        detect_numeric_annotation_types(emb_data)
    )

    print("Number of continuous variables:", len(continuous_variables))
    print("Number of binary variables:", len(binary_variables))
    print("Number of count variables:", len(count_variables))

    emb_data.annotations = add_class_specific_annotations(
        annotations=emb_data.annotations,
        targets=targets,
        continuous_variables=continuous_variables,
        count_variables=count_variables,
        binary_variables=binary_variables,
        class_names = args.class_names,
    )

    if args.is_numeric:
        numeric_variables += [
            f"{name}_{class_name}"
            for name in continuous_variables
            for class_name in args.class_names.values()
        ]

    train_data = Subset(emb_data, splits["train"])
    test_data = Subset(emb_data, splits["test"])
    val_data = Subset(emb_data, splits["val"])

    print("Preparing data loaders and model...")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    device = args.device if torch.cuda.is_available() else "cpu"

    sae_graph = ShallowSAE(
        in_dim=args.d_z,
        latent_dim=args.d_z * args.latent_multiplier,
        sparsity_weight=args.sparsity_weight
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path)

    sae_graph.load_state_dict(
        checkpoint
    )
    sae_graph = sae_graph.to(device)
    sae_graph.eval()

    print("Computing feature maxima from training data...")

    (
        max_features_train,
        frequency_stats_train,
        mean_active_activation_train,
    ) = compute_feature_maxima(
        sae_model=sae_graph,
        data_loader=train_loader,
        device=device,
    )

    print("Running GEA annotation on validation data...")
    f1_scores_concepts, concept_counts, frequency_stats_val, _, mean_active_activation_val, numeric_correlations = gea_annotation(
        sae_model = sae_graph, 
        data_loader = val_loader, 
        max_features=max_features_train,
        thresholds = args.thresholds, 
        device = device,
        numeric_variables=numeric_variables if args.is_numeric else None,
    )
    print("GEA annotation completed.")
    print("Selecting concept features...")
    concept_feature_pairs = select_concept_features(
        counts=concept_counts, 
        f1_scores=f1_scores_concepts,
        f1_threshold=args.f1_threshold,
        min_count=args.min_count
    )

    if args.is_numeric:
        print("Selecting numeric features...")
        numeric_feature_pairs = select_numeric_features(
            numeric_correlations=numeric_correlations,
            correlation_threshold=args.correlation_threshold,
        )

    print("Testing annotations...")
    test_results, numeric_feature_pairs_test = concept_feature_test(
        sae_model=sae_graph,
        data_loader=test_loader,
        max_features=max_features_train,
        concept_feature_pairs=concept_feature_pairs,
        numeric_feature_pairs=numeric_feature_pairs,
        numeric_variables=numeric_variables,
        device=device,
    )

    print("Computing test-set activation statistics...")
    max_features_test, frequency_stats_test, mean_active_activation_test = compute_feature_maxima(sae_model = sae_graph, 
                                                                                                  data_loader = test_loader,
                                                                                                  device = device)

    results = {"f1_scores_concepts": f1_scores_concepts, 
               "numeric_correlations": numeric_correlations,
               "concept_counts": concept_counts, 
               "frequency_stats_train": frequency_stats_train, 
               "frequency_stats_val": frequency_stats_val,
               "frequency_stats_test": frequency_stats_test,
               "max_features_train": max_features_train,
               "max_features_test": max_features_test,
               "mean_active_activation_train": mean_active_activation_train,
               "mean_active_activation_val": mean_active_activation_val,
               "mean_active_activation_test": mean_active_activation_test,
               "concept_feature_pairs": concept_feature_pairs,
               "numeric_feature_pairs": numeric_feature_pairs,
               "concept_feature_pairs_test": test_results,
               "numeric_feature_pairs_test": numeric_feature_pairs_test,
               "W_dec": sae_graph.W_dec.detach().cpu().numpy()}

    print("Saving results in:", args.results_path)
    torch.save(results, args.results_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="GEA annotation pipeline."
    )

    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="node_embeddings.npz",
        help="Path to npz file containing embeddings."
    )

    parser.add_argument(
        "--splits_path",
        type=str,
        default='splits.pt',
        help="Path with saved splits."
    )

    parser.add_argument(
        "--min_category_count",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of workers in data loader."
    )

    parser.add_argument(
        "--d_z",
        type=int,
        default=1200,
        help="Embedding vectors dimension."
    )

    parser.add_argument(
        "--latent_multiplier",
        type=int,
        default=8,
        help="Multiplier applied to input dimension to obtain SAE latent dimension."
    )

    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=1e-3,
        help="Weight of the sparsity penalty in the SAE loss."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device (cuda, cuda:0, cpu)."
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="sae_model.pt",
        help="Path with the trained SAE model weights."
    )

    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum number of counts per concept."
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0, 0.15, 0.5, 0.6, 0.8],
        help="One or more activation thresholds."
    )

    parser.add_argument(
        "--f1_threshold",
        type=float,
        default=0.5,
        help="Threshold for selecting features based on F1-score."
    )

    parser.add_argument(
        "--correlation_threshold",
        type=float,
        default=0.5,
        help="Minimum absolute correlation for selecting numeric features.",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="gea_annotation_results.pt",
        help="Path to save metrics."
    )

    parser.add_argument(
        "--is_numeric",
        action="store_true",
        help="Boolean flag indicating whether there are numeric annotations to binarise.",
    )

    parser.add_argument(
        "--class_names",
        type=json.loads,
        default={0: "insoluble", 1: "soluble"},
        help=(
            "JSON mapping of class labels to names, "
            'e.g. \'{"0": "insoluble", "1": "soluble"}\'.'
        ),
    )

    args = parser.parse_args()

    main(args)