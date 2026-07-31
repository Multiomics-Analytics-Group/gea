from gea.dataloader import EmbeddingDataset
from gea.analysis import gea_annotation, best_concept_features, concept_feature_test
from gea.gea import ShallowSAE
import torch
from torch.utils.data import DataLoader, Subset
import argparse

def main(args):

    emb_data = EmbeddingDataset(args.embeddings_path)

    splits = torch.load(args.splits_path)

    test_data = Subset(emb_data, splits["test"])
    val_data = Subset(emb_data, splits["val"])

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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sae_graph = ShallowSAE(
        in_dim=args.d_z,
        latent_dim=args.d_z * args.latent_multiplier,
        sparsity_weight=args.sparsity_weight
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path)

    sae_graph.load_state_dict(
        checkpoint["model_state_dict"]
    )
    sae_graph = sae_graph.to(device)

    best_features, concept_counts, frequency_stats, max_features = gea_annotation(
        sae_model = sae_graph, 
        data_loader = val_loader, 
        thresholds = args.thresholds, 
        top_k = args.top_k,
        device = device
    )

    concept_feature_pairs = best_concept_features(
        counts=concept_counts, 
        best_features=best_features,
        min_count=args.min_count
    )

    test_results = concept_feature_test(
        sae_model = sae_graph, 
        data_loader = test_loader, 
        max_features = max_features, 
        concept_feature_pairs = concept_feature_pairs, 
        device = device
    )

    torch.save(
        {
            "best_features": best_features,
            "concept_counts": concept_counts,
            "frequency_stats": frequency_stats,
            "max_features": max_features,
            "test_results": test_results
        },
        args.results_path
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="F1-scores computations for feature-concept pairs in the validation dataset."
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
        default=50,
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
        "--top_k",
        type=int,
        default=5,
        help="Number of top features to select during validation."
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="gea_annotation_results_test.pt",
        help="Path to save metrics."
    )

    args = parser.parse_args()

    main(args)