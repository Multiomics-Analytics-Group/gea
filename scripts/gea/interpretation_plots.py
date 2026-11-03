import os
import argparse
import torch
import json

from gea.results import (
    plot_feature_activity_upset,
    build_feature_annotations,
    plot_top_feature_annotations,
    plot_discriminative_feature_proportions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse GEA results and generate SAE feature annotation plots."
    )

    # Input files
    parser.add_argument(
        "--gea_result_path",
        type=str,
        required=True,
        help="Path to the GEA results obtained with gea_pipeline.py.",
    )
    parser.add_argument(
        "--sae_annotation_association_path",
        type=str,
        required=True,
        help="Path to the SAE annotation association results obtained with sae_annotation_association.py.",
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gea_results",
        help="Name or path of the directory where all generated plots will be saved (default: gea_results).",
    )

    # Feature annotation thresholds
    parser.add_argument(
        "--min_categorical_mean",
        type=float,
        default=0.60,
        help="Minimum mean association score required for a categorical feature annotation (default: 0.60).",
    )
    parser.add_argument(
        "--max_categorical_gap",
        type=float,
        default=0.20,
        help="Maximum allowed difference between validation and test association scores for categorical annotations (default: 0.20).",
    )
    parser.add_argument(
        "--min_categorical_test",
        type=float,
        default=0.50,
        help="Minimum test association score required for a categorical feature annotation (default: 0.50).",
    )
    parser.add_argument(
        "--min_numeric_mean",
        type=float,
        default=0.50,
        help="Minimum mean association score required for a numeric feature annotation (default: 0.50).",
    )
    parser.add_argument(
        "--max_numeric_gap",
        type=float,
        default=0.20,
        help="Maximum allowed difference between validation and test association scores for numeric annotations (default: 0.20).",
    )
    parser.add_argument(
        "--min_numeric_test",
        type=float,
        default=0.40,
        help="Minimum test association score required for a numeric feature annotation (default: 0.40).",
    )

    # Feature annotation plots
    parser.add_argument(
        "--top_k_annotations",
        type=int,
        default=30,
        help="Number of top categorical or numeric feature annotations to display in the corresponding plot (default: 30).",
    )
    parser.add_argument(
        "--is_numeric",
        action="store_true",
        help="Generate the numeric feature annotation plot in addition to the categorical feature annotation plot.",
    )

    # Discriminative feature proportions
    parser.add_argument(
        "--class_names",
        type=json.loads,
        default=None,
        help='Mapping between class labels and class names, provided as a JSON dictionary (e.g. \'{"0": "insoluble", "1": "soluble"}\'). Defaults to {0: "class_0", 1: "class_1"}.',
    )
    parser.add_argument(
        "--positive_label",
        type=int,
        default=1,
        help="Label identifying the positive class used in the discriminative feature analysis (default: 1).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top discriminative features to include in the feature proportion plots (default: 10).",
    )
    parser.add_argument(
        "--probability_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to assign predicted class labels when evaluating feature activations (default: 0.5).",
    )
    parser.add_argument(
        "--min_active",
        type=int,
        default=10,
        help="Minimum number of active samples required for a feature to be included in the discriminative feature analysis (default: 10).",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(0, 1),
        metavar=("YMIN", "YMAX"),
        help="Lower and upper limits of the y-axis in the discriminative feature proportion plots (default: 0 1).",
    )

    return parser.parse_args()


def main(args):

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GEA results
    gea_result = torch.load(args.gea_result_path, weights_only=False)

    # Plot feature activity
    plot_feature_activity_upset(
        gea_result,
        feature_key="active_features",
        title="Active features across data splits",
        output_path=os.path.join(
            args.output_dir, "feature_activity_upset.png"
        ),
        figsize=(9, 5),
    )

    # Build feature annotations
    feature_annotations = build_feature_annotations(
        gea_result["concept_feature_pairs"],
        gea_result["numeric_feature_pairs"],
        min_categorical_mean=args.min_categorical_mean,
        max_categorical_gap=args.max_categorical_gap,
        min_categorical_test=args.min_categorical_test,
        min_numeric_mean=args.min_numeric_mean,
        max_numeric_gap=args.max_numeric_gap,
        min_numeric_test=args.min_numeric_test,
    )

    # Plot categorical annotations
    plot_top_feature_annotations(
        feature_annotations,
        annotation_type="categorical",
        title=None,
        top_k=args.top_k_annotations,
        path=os.path.join(
            args.output_dir, "top_categorical_features.png"
        ),
    )

    # Plot numeric annotations
    if args.is_numeric:
        plot_top_feature_annotations(
            feature_annotations,
            annotation_type="numeric",
            title=None,
            top_k=args.top_k_annotations,
            path=os.path.join(
                args.output_dir, "top_numeric_features.png"
            ),
        )

    # Load SAE annotation associations
    sae_annotation_association = torch.load(
        args.sae_annotation_association_path,
        weights_only=False,
    )

    # Plot discriminative feature proportions
    class_names = (
        {int(label): name for label, name in args.class_names.items()}
        if args.class_names is not None
        else None
    )
    plot_discriminative_feature_proportions(
        class_activations=sae_annotation_association,
        feature_annotations=feature_annotations,
        class_names=class_names,
        positive_label=args.positive_label,
        top_k=args.top_k,
        probability_threshold=args.probability_threshold,
        accepted_only=True,
        figsize_per_feature=(5, 3.5),
        output_path=os.path.join(
            args.output_dir, "discriminative_feature_proportions.png"
        ),
        threshold_key="threshold",
        activation_thresholds=(0, 0.15, 0.5, 0.6, 0.8),
        min_active=args.min_active,
        ylim=args.ylim,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)