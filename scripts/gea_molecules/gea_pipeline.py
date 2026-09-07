from gea.dataloader import EmbeddingDataset
from gea.analysis import gea_annotation, concept_feature_test, select_concept_features, calculate_quartile_thresholds, binarize_annotation
from gea.gea import ShallowSAE
import torch
from torch.utils.data import DataLoader, Subset
import argparse
from gea_molecules.results import selected_feature_activation

def main(args):

    emb_data = EmbeddingDataset(args.embeddings_path)

    splits = torch.load(args.splits_path, weights_only=False)
    
    train_indices = splits["train"]

    if args.is_graph:
        quartile_thresholds = calculate_quartile_thresholds(
            emb_data,
            train_indices
        )

        for idx in range(len(emb_data)):

            emb_data.annotations[idx] = binarize_annotation(
                emb_data.annotations[idx],
                quartile_thresholds
            )

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


    f1_scores_concepts, concept_counts, frequency_stats, max_features = gea_annotation(
        sae_model = sae_graph, 
        data_loader = val_loader, 
        thresholds = args.thresholds, 
        device = device
    )

    concept_feature_pairs = select_concept_features(
        counts=concept_counts, 
        f1_scores=f1_scores_concepts,
        f1_threshold=args.f1_threshold,
        min_count=args.min_count
    )


    test_results = concept_feature_test(
        sae_model = sae_graph, 
        data_loader = test_loader, 
        max_features = max_features, 
        concept_feature_pairs = concept_feature_pairs, 
        device = device
    )

    # Extract activations for all features
    
    acts = selected_feature_activation(sae_graph, test_loader, max_features=max_features, feature=2789, device='cuda')

    results = {"f1_scores_concepts": f1_scores_concepts, 
               "concept_counts": concept_counts, 
               "frequency_stats": frequency_stats, 
               "max_features": max_features,
               "concept_feature_pairs": concept_feature_pairs,
               "test_results": test_results,
               "W_dec": sae_graph.W_dec.detach().cpu().numpy()}

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
        "--f1_threshold",
        type=float,
        default=0.5,
        help="Threshold for selecting features based on F1-score."
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="gea_annotation_results.pt",
        help="Path to save metrics."
    )

    parser.add_argument(
        "--is_graph",
        action="store_true",
        help="Annotate graph embeddings instead of node embeddings."
    )

    args = parser.parse_args()

    main(args)