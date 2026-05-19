"""
Extract ESM-2 embeddings for a list of Ensembl gene IDs.

Produces a file compatible with gea.utils.load_esm_embeddings:
    { ensembl_id: { transcript_id: tensor([hidden_dim]) } }

Isoform handling: all unique protein sequences for each gene are embedded.
Aggregation (mean / max across isoforms) happens at load time in load_esm_embeddings,
so this script stores all isoforms to keep the file maximally flexible.

Usage
-----
    python scripts/get_esm_embeddings.py \\
        --gene-list data/ensembl_ids.txt \\
        --out      data/esm_embeddings.pt \\
        --species  human \\
        --model    facebook/esm2_t33_650M_UR50D \\
        --batch-size 8

The gene list file should contain one Ensembl gene ID (ENSG...) per line.
It is generated automatically by the preprocessing pipeline:

    import pandas as pd
    ensembl_ids = list(filter_counts_ppi.index)
    with open("data/ensembl_ids.txt", "w") as f:
        f.write("\\n".join(ensembl_ids))
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

parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings for GEA.")
parser.add_argument(
    "--gene-list",
    required=True,
    help="Path to a text file with one Ensembl gene ID per line.",
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
    help="Fraction of genes to use (for testing, e.g. 0.001).",
)
args = parser.parse_args()

# ── Load gene list ─────────────────────────────────────────────────────────────

with open(args.gene_list) as f:
    target_genes = {line.strip() for line in f if line.strip()}

print(f"Loaded {len(target_genes)} target Ensembl gene IDs.")

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

# ── Parse sequences — keep all unique isoforms per gene ───────────────────────

print("Parsing sequences and resolving isoforms...")
gene_to_seq = {}  # { ensembl_gene_id: { transcript_id: sequence } }

with gzip.open(fasta_path, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        desc = record.description
        if "gene:" not in desc or "transcript:" not in desc:
            continue

        gene_block = next((t for t in desc.split() if t.startswith("gene:")), None)
        tx_block = next((t for t in desc.split() if t.startswith("transcript:")), None)
        if gene_block is None or tx_block is None:
            continue

        gene_id = gene_block.replace("gene:", "").split(".")[0]
        tx_id = tx_block.replace("transcript:", "").split(".")[0]

        if gene_id not in target_genes:
            continue

        seq_str = str(record.seq)
        gene_to_seq.setdefault(gene_id, {})
        # Only add if this exact sequence hasn't been seen for this gene
        if seq_str not in gene_to_seq[gene_id].values():
            gene_to_seq[gene_id][tx_id] = seq_str

n_isoforms = sum(len(v) for v in gene_to_seq.values())
print(
    f"Mapped {len(gene_to_seq)}/{len(target_genes)} genes "
    f"to {n_isoforms} unique isoforms."
)

# ── Optional subset for testing ───────────────────────────────────────────────

if args.subset_frac is not None:
    k = max(1, int(args.subset_frac * len(gene_to_seq)))
    subset_keys = random.sample(list(gene_to_seq.keys()), k)
    gene_to_seq = {g: gene_to_seq[g] for g in subset_keys}
    print(f"Using subset of {len(gene_to_seq)} genes for testing.")

# ── Load ESM-2 model ───────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {args.model} onto {device}...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = EsmModel.from_pretrained(args.model)
model.to(device)
model.eval()

# ── Flatten isoforms for batched inference ─────────────────────────────────────

flat = [
    {"gene_id": g, "transcript_id": t, "sequence": s}
    for g, transcripts in gene_to_seq.items()
    for t, s in transcripts.items()
]
print(f"Computing embeddings for {len(flat)} isoforms in batches of {args.batch_size}…")

nested_embeddings = {g: {} for g in gene_to_seq}

for i in tqdm(range(0, len(flat), args.batch_size)):
    batch = flat[i : i + args.batch_size]
    seqs = [item["sequence"] for item in batch]

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

    for j, item in enumerate(batch):
        nested_embeddings[item["gene_id"]][item["transcript_id"]] = cls_emb[j]

# ── Save ───────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(nested_embeddings, args.out)
print(f"Saved embeddings to {args.out}")
print(
    f"Format: {{ensembl_id: {{transcript_id: tensor([{cls_emb.shape[1]}])}}}}  "
    f"({len(nested_embeddings)} genes)"
)
