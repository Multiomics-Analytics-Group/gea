"""
Assign each disease in a DISEASES-derived table (identified by DOID) to a broad category —
cancer, infectious, immune, nervous, ... — read off the Disease Ontology's own is_a hierarchy.

Usage
-----
    python docs/scripts/build_disease_categories.py \\
        --diseases docs/tutorial/data/diseases/diseases_filtered.tsv \\
        --out      docs/tutorial/data/diseases/disease_categories.tsv

Output is a TSV with one row per distinct DOID in --diseases: doid, doid_name, category.
"""

import argparse
import os
import urllib.request
from pathlib import Path

import pandas as pd

# (category, root DOID) — the first match down this list wins, which is what settles diseases
# sitting under two branches at once: a lung carcinoma is filed under "cancer" rather than
# "respiratory", multiple sclerosis under "immune" rather than "nervous". The last few are DO's
# cross-cutting branches and catch what it files under no organ system at all.
DO_CATEGORY_ROOTS = [
    ("cancer",           "DOID:14566"),    # disease of cellular proliferation
    ("infectious",       "DOID:0050117"),  # disease by infectious agent
    ("immune",           "DOID:2914"),
    ("nervous",          "DOID:863"),
    ("mental",           "DOID:150"),      # disease of mental health
    ("cardiovascular",   "DOID:1287"),
    ("metabolic",        "DOID:0014667"),  # disease of metabolism
    ("endocrine",        "DOID:28"),
    ("respiratory",      "DOID:1579"),
    ("gastrointestinal", "DOID:77"),
    ("hematopoietic",    "DOID:74"),
    ("urinary",          "DOID:18"),
    ("reproductive",     "DOID:15"),
    ("musculoskeletal",  "DOID:17"),
    ("integumentary",    "DOID:16"),
    ("syndrome",         "DOID:225"),
    ("physical",         "DOID:0080015"),  # physical disorder: congenital malformations
    ("genetic",          "DOID:630"),
]

DO_OBO_URL = ("https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/"
              "main/src/ontology/HumanDO.obo")


class DiseaseOntology:
    """The is_a hierarchy of the Human Disease Ontology, read from the .obo release."""

    def __init__(self, path):
        self.parents, self.names = {}, {}
        self._ancestors = {}
        term = None
        for line in Path(path).read_text().splitlines():
            if line.startswith("["):  # a new stanza closes the previous term
                if term and term["id"]:
                    self.parents[term["id"]] = term["is_a"]
                    self.names[term["id"]] = term["name"]
                term = {"id": None, "name": None, "is_a": []} if line == "[Term]" else None
            elif term is not None and line:
                if line.startswith("id: "):
                    term["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    term["name"] = line[6:].strip()
                elif line.startswith("is_a: "):
                    term["is_a"].append(line[6:].split("!")[0].strip())
        if term and term["id"]:
            self.parents[term["id"]] = term["is_a"]
            self.names[term["id"]] = term["name"]

    def ancestors(self, doid):
        if doid not in self._ancestors:
            seen, stack = set(), list(self.parents.get(doid, ()))
            while stack:  # DO is a DAG, so this is not a walk up a single chain
                parent = stack.pop()
                if parent not in seen:
                    seen.add(parent)
                    stack.extend(self.parents.get(parent, ()))
            self._ancestors[doid] = seen
        return self._ancestors[doid]

    def category(self, doid):
        if doid not in self.parents:
            return "other"
        lineage = self.ancestors(doid) | {doid}
        return next((cat for cat, root in DO_CATEGORY_ROOTS if root in lineage), "other")


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diseases", required=True, help="Path to diseases_filtered.tsv (needs doid, doid_name).")
    parser.add_argument("--out", required=True, help="Output TSV path: doid, doid_name, category.")
    parser.add_argument("--obo", default=None, help="Path to HumanDO.obo (downloaded next to --out if omitted).")
    args = parser.parse_args()

    obo_path = args.obo or os.path.join(os.path.dirname(os.path.abspath(args.out)), "HumanDO.obo")
    download_file(DO_OBO_URL, obo_path)
    do = DiseaseOntology(obo_path)
    print(f"Disease Ontology: {len(do.parents):,} terms")

    diseases = pd.read_csv(args.diseases, sep="\t")
    universe = diseases[["doid", "doid_name"]].drop_duplicates("doid").reset_index(drop=True)

    universe["category"] = universe["doid"].map(do.category)
    unknown = (universe["category"] == "other") & ~universe["doid"].isin(do.parents)
    if unknown.any():
        print(f"{unknown.sum()} DOIDs not found in this DO release, filed under 'other'")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    universe.to_csv(args.out, sep="\t", index=False)

    print(f"\n{len(universe):,} diseases across {universe['category'].nunique()} categories")
    print(universe["category"].value_counts().to_string())
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
