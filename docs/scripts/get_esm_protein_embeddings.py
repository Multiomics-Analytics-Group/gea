"""
Extract ESM-2 embeddings for a list of Ensembl/STRING protein IDs.

Unlike get_esm_embeddings.py (which is keyed by Ensembl *gene* id and aggregates over every
isoform of that gene), this script is keyed directly by Ensembl *protein* id (ENSP...) — the
same identifier STRING uses for its network nodes. Each protein id already names one specific
sequence, so there is no isoform aggregation to do: the output is a flat mapping,

    { protein_id: tensor([hidden_dim]) }

Usage
-----
    python scripts/get_esm_protein_embeddings.py \\
        --protein-list data/ensembl_protein_ids.txt \\
        --out          data/esm_protein_embeddings.pt \\
        --species      human \\
        --model        facebook/esm2_t33_650M_UR50D \\
        --batch-size 8

The protein list file should contain one Ensembl protein ID (ENSP...) per line.
"""

import argparse
import gzip
import os
import random
import urllib.request

import torch
from Bio import SeqIO
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings for GEA, keyed by protein id.")
parser.add_argument(
    "--protein-list",
    required=True,
    help="Path to a text file with one Ensembl protein ID (ENSP...) per line.",
)
parser.add_argument(
    "--out",
    required=True,
    help="Output .pt file path.",
)
parser.add_argument(
    "--species",
    default="human",
    choices=["human"],
    help="Species (currently only human supported).",
)
parser.add_argument(
    "--model",
    default="facebook/esm2_t33_650M_UR50D",
    help="HuggingFace model ID for ESM-2 (default: 650M).",
)
parser.add_argument(
    "--fasta",
    default=None,
    help=(
        "Path to Ensembl proteome FASTA (.fa.gz). "
        "Downloaded automatically if not provided."
    ),
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="Number of sequences per GPU batch (default: 8).",
)
parser.add_argument(
    "--max-length",
    type=int,
    default=1024,
    help="Maximum sequence length passed to the tokenizer (default: 1024).",
)
parser.add_argument(
    "--subset-frac",
    type=float,
    default=None,
    help="Fraction of proteins to use (for testing, e.g. 0.001).",
)
args = parser.parse_args()

# ── Load protein list ───────────────────────────────────────────────────────────

with open(args.protein_list) as f:
    target_proteins = {line.strip() for line in f if line.strip()}

print(f"Loaded {len(target_proteins)} target Ensembl protein IDs.")

# ── Ensembl proteome FASTA ─────────────────────────────────────────────────────

ENSEMBL_URLS = {
    "human": "https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz",
}

if args.fasta is None:
    fasta_path = os.path.join(os.path.dirname(args.out), "ensembl_pep.fa.gz")
else:
    fasta_path = args.fasta

if not os.path.exists(fasta_path):
    url = ENSEMBL_URLS[args.species]
    print(f"Downloading Ensembl proteome for {args.species} from:\n  {url}")
    urllib.request.urlretrieve(url, fasta_path)
    print("Download complete.")
else:
    print(f"Using existing proteome FASTA: {fasta_path}")

# ── Parse sequences — one sequence per protein id, no isoform grouping ────────

print("Parsing sequences...")
protein_to_seq = {}  # { ensembl_protein_id: sequence }

with gzip.open(fasta_path, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        protein_id = record.id.split(".")[0]  # record.id is the ENSP id itself, e.g. ENSP00000451042.1
        if protein_id not in target_proteins:
            continue
        protein_to_seq[protein_id] = str(record.seq)

print(f"Mapped {len(protein_to_seq)}/{len(target_proteins)} proteins to a sequence.")

# ── Optional subset for testing ───────────────────────────────────────────────

if args.subset_frac is not None:
    k = max(1, int(args.subset_frac * len(protein_to_seq)))
    subset_keys = random.sample(list(protein_to_seq.keys()), k)
    protein_to_seq = {p: protein_to_seq[p] for p in subset_keys}
    print(f"Using subset of {len(protein_to_seq)} proteins for testing.")

# ── Load ESM-2 model ───────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {args.model} onto {device}...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = EsmModel.from_pretrained(args.model)
model.to(device)
model.eval()

# ── Batched inference ──────────────────────────────────────────────────────────

flat = list(protein_to_seq.items())
print(f"Computing embeddings for {len(flat)} proteins in batches of {args.batch_size}…")

embeddings = {}

for i in tqdm(range(0, len(flat), args.batch_size)):
    batch = flat[i : i + args.batch_size]
    seqs = [seq for _, seq in batch]

    inputs = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        # CLS token (position 0) as the sequence-level representation
        cls_emb = out.last_hidden_state[:, 0, :].cpu()

    for j, (protein_id, _) in enumerate(batch):
        embeddings[protein_id] = cls_emb[j]

# ── Save ───────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(embeddings, args.out)
print(f"Saved embeddings to {args.out}")
print(f"Format: {{protein_id: tensor([{cls_emb.shape[1]}])}}  ({len(embeddings)} proteins)")
