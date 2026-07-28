import argparse
from gea.dataloader import EmbeddingDataset
from gea.gea import ShallowSAE, train_sae
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

def main(args):

    torch.manual_seed(args.seed)

    emb_data = EmbeddingDataset(args.embeddings_path)
    g = torch.Generator().manual_seed(args.seed)
    train_pct, val_pct, test_pct = args.splits

    if not np.isclose(sum(args.splits), 1.0):
        raise ValueError(
            f"Splits must sum to 1. Got {args.splits}"
        )

    n = len(emb_data)
    train_size = int(train_pct * n)
    val_size = int(val_pct * n)
    test_size = n - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        emb_data,
        [train_size, val_size, test_size],
        generator=g,
    )

    torch.save(
        {
            "train": train_data.indices,
            "val": val_data.indices,
            "test": test_data.indices,
        },
        args.splits_path,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Dataset size: {n}")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    print(f"Device: {device}")
    print(
        f"SAE dimensions: {args.d_z} -> "
        f"{args.d_z * args.latent_multiplier}"
    )

    sae_graph = ShallowSAE(
        in_dim=args.d_z,
        latent_dim=args.d_z * args.latent_multiplier,
        sparsity_weight=args.sparsity_weight
    ).to(device)

    train_sae(
        sae_model=sae_graph,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        w_l2=args.weight_decay,
    )

    torch.save(
        {
            "model_state_dict": sae_graph.state_dict(),
            "config": vars(args),
        },
        args.checkpoint_path,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Annotate Grover molecular embeddings."
    )

    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="node_embeddings.npz",
        help="Path to npz file containing embeddings."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for data split."
    )

    parser.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/validation/test split fractions"
    )

    parser.add_argument(
        "--splits_path",
        type=str,
        default='splits.pt',
        help="Output pt path to save splits."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size."
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
        help="Multiplier applied to input dimension to obtain SAE latent dimension"
    )

    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=1e-3,
        help="Weight of the sparsity penalty in the SAE loss"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="sae_model.pt",
        help="Path where the trained SAE model will be saved"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device (cuda, cuda:0, cpu)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    main(args)
