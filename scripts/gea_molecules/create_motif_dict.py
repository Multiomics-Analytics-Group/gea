from rdkit.Chem import FunctionalGroups
import pickle
from pathlib import Path
import csv
from rdkit import RDConfig
from collections import defaultdict

directory = Path("dict")
directory.mkdir(parents=True, exist_ok=True)

motifs = {}
fg_tree = FunctionalGroups.BuildFuncGroupHierarchy()

for fg in fg_tree:
    motifs[fg.name] = fg.smarts
    subfg = fg.children
    for child in subfg:
        motifs[fg.name + " - " + child.name] = child.smarts

def load_fragment_motifs(motifs):
    rdkit_data = Path(RDConfig.RDDataDir)

    matches = list(rdkit_data.rglob("FragmentDescriptors.csv"))

    if not matches:
        raise FileNotFoundError(
            "Could not find FragmentDescriptors.csv in the RDKit installation."
        )

    csv_path = matches[0]

    with open(csv_path, "r", encoding="utf-8") as f:
        # The header starts with '#', so remove '#' only from the header.
        lines = []
        for line in f:
            if line.startswith("#Code"):
                line = line[1:]
            elif line.lstrip().startswith("#"):
                continue

            if line.strip():
                lines.append(line)

        reader = csv.DictReader(lines, delimiter="\t")

        for row in reader:
            name = row.get("Description")
            smarts = row.get("SMARTS")

            if not name or not smarts:
                continue

            name = name.strip().strip('"')
            smarts = smarts.strip().strip('"')

            if name.lower().startswith("number of "):
                name = name[len("Number of "):]

            name = name[0].upper() + name[1:]

            motifs[name] = smarts

    return motifs

motifs = load_fragment_motifs(motifs)

motifs['Amine - Cyclic'] = "[N;R;$(N-[#6]);!$(N=*);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]"

# General carbon environments
motifs["C-H3"] = "[CH3]"
motifs["C-H2"] = "[CH2]"
motifs["C-H1"] = "[CH]"
motifs["C-H0"] = "[C;H0]"

motifs["Aromatic carbon"] = "[c]"
motifs["Aliphatic carbon"] = "[C;A]"
motifs["Cyclic carbon"] = "[C;R]"
motifs["Acyclic carbon"] = "[C;!R]"

motifs["Aromatic cyclic carbon"] = "[c;R]"
motifs["Aliphatic cyclic carbon"] = "[C;A;R]"
motifs["Aliphatic acyclic carbon"] = "[C;A;!R]"

#motifs["Aromatic CH"] = "[cH]"
#motifs["Aromatic C substituted"] = "[cH0]"

motifs['Aromatic CH adjacent to substituted carbon'] = '[cH](:[cH0])'
motifs['Aromatic carbon attached to aliphatic carbon'] = '[c]-[C;!a]'
motifs['Aromatic carbon attached to heteroatom'] = '[c]-[!#6;!#1]'

# Hybridization
motifs["C sp3"] = "[C;X4]"
motifs["C sp2"] = "[C;X3]"
motifs["C sp"] = "[C;X2]"

# Alkane-specific
motifs["Alkane"] = "[C;X4;!a]"
motifs["Alkane CH3"] = "[CH3;X4]"
motifs["Alkane CH2"] = "[CH2;X4]"
motifs["Alkane CH"] = "[CH;X4]"
motifs["Alkane C"] = "[C;H0;X4]"

# Aromatic
motifs["Benzene ring"] = "c1ccccc1"

# Amides
motifs["Amide"] = "[CX3](=O)[NX3]"
motifs["Amide - Primary"] = "[CX3](=O)[NH2]"
motifs["Amide - Secondary"] = '[CX3](=O)[NH1]([#6])'
motifs["Amide - Tertiary"] = '[CX3](=O)[N;H0]([#6])[#6]'
motifs["Lactam"] = '[C;R](=O)[N;R]'
motifs["Amide - Aromatic"] = '[c][CX3](=O)[NX3]'
motifs["Amide - N-Aromatic"] = '[CX3](=O)[N;X3]-c'

# alkene
motifs["Alkene"] = '[C;X3;!a]=[C;X3;!a]'
motifs["Alkene - terminal"] = '[C;X3;!a]=[CH2]'
motifs["Alkene - internal"] = '[C;X3;!a]([#6])=[C;X3;!a]([#6])'
motifs["Alkene - conjugated"] = '[C;X3;!a]=[C;X3;!a]-[C;X3;!a]=[C;X3;!a]'

motifs["Alkene CH2"] = "[C;H2]=[*]"
motifs["Alkene CH"] = "[C;H1]=[*]"
motifs["Alkene C"] = "[C;H0]=[*]"

# Alkynes
motifs["Alkyne"] = "[CX2]#[CX2]"
motifs["Alkyne - terminal"] = '[C;X2]#[CH]'
motifs["Alkyne - internal"] = '[C;X2]([#6])#[C;X2]([#6])'

# Ether
motifs["Ether"] = '[OD2]([#6])[#6]'
motifs["Ether - aliphatic"] = '[O;D2]([C;!a])[C;!a]'
motifs["Ether - aromatic"] = '[O;D2]([c])[#6]'
motifs["Ether - diaryl"] = '[O;D2](c)c'
motifs["Ether - cyclic"] = '[O;R;D2]'
motifs["Epoxide"] = "[OX2r3]([#6r3])[#6r3]"
motifs["Ether - 5-membered ring"] = '[O;R;r5]'
motifs["Ether - 6-membered ring"] = '[O;R;r6]'

# Thiol
motifs["Thiol"] = "[SX2H]"
motifs["Thiol - aliphatic"] = '[S;H1;$(S-[C;!a])]'
motifs["Thiol - aromatic"] = '[S;H1;$(S-c)]'

# Ketone
motifs["Ketone"] = "[#6][CX3](=O)[#6]"
motifs["Ketone - aliphatic"] = '[C;!a][CX3](=O)[C;!a]'
motifs["Ketone - aromatic"] = '[c][CX3](=O)[#6]'
motifs["Ketone - cyclic"] = '[C;R][CX3;R](=O)[C;R]'
motifs["Ketone - alpha,beta-unsaturated"] = '[C;X3;!a]=[C;X3;!a][CX3](=O)[#6]'

# Ester
motifs["Ester"] = "[CX3](=O)[OX2H0][#6]"
motifs["Ester - aliphatic"] = '[C;!a][CX3](=O)[OX2H0][C;!a]'
motifs["Ester - aromatic carbonyl"] = '[c][CX3](=O)[OX2H0][#6]'
motifs["Ester - aryl oxygen"] = '[CX3](=O)[OX2H0][c]'
motifs["Lactone"] = '[C;R](=O)[O;R]'

# Nitrile
motifs["Nitrile"] = "[CX2]#[NX1]"
motifs["Nitrile - aliphatic"] = '[C;X2;!a]#[N;X1]'
motifs["Nitrile - aromatic"] = '[c]-[C;X2]#[N;X1]'
motifs["Nitrile - cyclic"] = '[C;X2;R]#[N;X1]'

# Imine
motifs["Imine"] = "[CX3]=[NX2]"
motifs["Cyclic imine"] = "[C;R]=[N;R]"
motifs["Aldimine"] = "[CH;X3]=[NX2]"
motifs["Ketimine"] = "[C;X3;H0]=[NX2]"


# Anhydride
motifs["Anhydride"] = "[CX3](=O)[OX2][CX3](=O)"
motifs["Cyclic anhydride"] = "[C;R](=O)[O;R][C;R](=O)"

# Sulfide
motifs["Sulfide"] = "[SX2]([#6])[#6]"
motifs["Sulfide - aliphatic"] = "[S;X2]([C;!a])[C;!a]"
motifs["Sulfide - aromatic"] = "[S;X2](c)[#6]"
motifs["Sulfide - diaryl"] = "[S;X2](c)c"
motifs["Sulfide - cyclic"] = "[S;X2;R]"

# Sulfonamide
motifs["Sulfonamide"] = '[S](=O)(=O)([#6])[NH]'
motifs["Sulfonamide - primary"] = '[S](=O)(=O)([#6])[NH2]'
motifs["Sulfonamide - secondary"] = '[S](=O)(=O)([#6])[NH1]([#6])'
motifs["Sulfonamide - tertiary"] = '[S](=O)(=O)([#6])[N;H0]([#6])[#6]'

# Extra halogens
motifs['Halogen - Fluorine'] = '[F]'
motifs['Halogen - Chlorine'] = '[Cl]'
motifs['Halogen - Iodine'] = '[I]'
motifs['Fluorine - Aromatic'] = '[F;$(*-!@c)]'
motifs['Fluorine - Aliphatic'] = '[F;$(*-!@C)]'

# General annotations
motifs["Aromatic atom"] = "[a]"
motifs["Aliphatic atom"] = "[A]"
motifs["Cyclic atom"] = "[R]"
motifs["Acyclic atom"] = "[!R]"

motifs["Aromatic cyclic atom"] = "[a;R]"
motifs["Aliphatic cyclic atom"] = "[A;R]"
motifs["Aliphatic acyclic atom"] = "[A;!R]"

# Number of Hs attached to the atom
motifs["H0"] = "[H0]"
motifs["H1"] = "[H1]"
motifs["H2"] = "[H2]"
motifs["H3"] = "[H3]"
motifs["H4"] = "[H4]"

# number of heavy-atom neighbors
motifs["degree 0"] = "[D0]"
motifs["degree 1"] = "[D1]"
motifs["degree 2"] = "[D2]"
motifs["degree 3"] = "[D3]"
motifs["degree 4"] = "[D4]"

# ============================================================
# Solubility model-specific atom environments
# Based on the atom typing rules from the solubility model
# ============================================================

# ------------------------------------------------------------
# Aliphatic carbon environments
# ------------------------------------------------------------

# sp3 carbon with different numbers of attached hydrogens
motifs["sp3 CH4"] = "[CX4;H4]"
motifs["sp3 CH3"] = "[CX4;H3]"
motifs["sp3 CH2"] = "[CX4;H2]"
motifs["sp3 CH"] = "[CX4;H1]"
motifs["sp3 C"] = "[CX4;H0]"

# sp3 carbon attached to another carbon
motifs["sp3 CH3 attached to carbon"] = "[CX4;H3][#6]"
motifs["sp3 CH2 attached to carbon"] = "[CX4;H2][#6]"
motifs["sp3 CH attached to two carbons"] = "[CX4;H]([#6])[#6]"
motifs["sp3 C attached to three carbons"] = "[CX4;H0]([#6])([#6])[#6]"
motifs["sp3 C attached to four carbons"] = "[CX4;H0]([#6])([#6])([#6])[#6]"

# More general tertiary/quaternary carbon environments
motifs["Tertiary sp3 carbon"] = "[CX4;H1;D3]"
motifs["Quaternary sp3 carbon"] = "[CX4;H0;D4]"

# ------------------------------------------------------------
# sp3 carbon attached to heteroatoms
# ------------------------------------------------------------

motifs["sp3 CH3 attached to O or N"] = \
    "[CX4;H3][CX3,c,F,Cl,Br,I]=[#8,#7]"

motifs["sp3 CH3 attached to heteroatom"] = \
    "[CX4;H3][CX4,c,F,Cl,Br,I]~[#8,#7]"

motifs["sp3 CH2 attached to carbonyl"] = \
    "[CX4;H2][#6]=[#8,#7]"

motifs["sp3 CH2 attached to heteroatom"] = \
    "[CX4;H2][CX4,c,F,Cl,Br,I]~[#8,#7]"

motifs["sp3 CH2 attached to OH or NH"] = \
    "[CX4;H2]-[OH,NH2,NH]"

motifs["sp3 CH attached to OH or NH"] = \
    "[CX4;H1]-[OH,NH2,NH]"

# ------------------------------------------------------------
# Alkyl branching
# ------------------------------------------------------------

motifs["Isopropyl-like carbon"] = "C([#6])([#6])"
motifs["Tert-butyl-like carbon"] = "C([#6])([#6])([#6])"

# ------------------------------------------------------------
# Unsaturated carbon environments
# ------------------------------------------------------------

motifs["Carbonyl CH"] = "[C;H1]=O"
motifs["Carbonyl C"] = "[C;H0]=O"

# Carbonyl carbon in a ring
motifs["Cyclic carbonyl"] = "[C;r]=O"

# ------------------------------------------------------------
# Aromatic carbon environments
# ------------------------------------------------------------

motifs["Aromatic CH"] = "[c;H1]"
motifs["Aromatic C substituted"] = "[c;H0]"

motifs["Aromatic CH in six-membered ring"] = "[c;H1;r6]"

# Aromatic CH connected to aromatic carbons
motifs["Aromatic CH connected to aromatic carbons"] = \
    "[c;H1](~c)~c"

# Aromatic CH connected to heteroaromatic atom
motifs["Aromatic CH adjacent to heteroatom"] = \
    "[c;H1](~c)~[a;!c]"

motifs["Aromatic CH between heteroatoms"] = \
    "[c;H1](~[a;!c])~[a;!c]"

# Aromatic substituted carbon connected to heteroatoms
motifs["Aromatic C adjacent to heteroatoms"] = \
    "[c;H0](~[a;!c])~[a;!c]"

motifs["Aromatic C attached to heteroatom and carbon"] = \
    "[c;H0](~[!#6])(~c)~[a;!c]"

# Aromatic substituted carbon in an all-carbon aromatic environment
motifs["Aromatic substituted C in carbon ring"] = \
    "[c;H0](~c)(~c)~c"

# Aromatic substituted carbon attached to aliphatic carbon/halogen
motifs["Aromatic C attached to aliphatic or halogen"] = \
    "[c;H0](~[CX4,F,Cl,Br,I])(~c)~c"

# Aromatic carbon attached to non-carbon/non-halogen atom
motifs["Aromatic C attached to heteroatom"] = \
    "[c;H0](~[!#6;!F;!Cl])(~c)~c"

# ------------------------------------------------------------
# Oxygen environments
# ------------------------------------------------------------

motifs["Hydroxyl O"] = "[#8;H1]"
motifs["Hydroxyl O attached to carbon"] = "[#8;H1]C"

motifs["Non-hydroxyl O"] = "[#8;H0]"
motifs["O attached to carbon and oxygen"] = "[#8;H0]CO"

motifs["Dialkyl O"] = "[#8;H0]([#6X4])[#6X4]"

motifs["Carbonyl O"] = "[#8]=C"
motifs["Carbonyl O attached to carbonyl carbon"] = "[#8]=C([#6])[#6]"

# Aromatic oxygen
motifs["Aromatic oxygen"] = "[o]"

# ------------------------------------------------------------
# Phosphorus / sulfur oxygen environments
# ------------------------------------------------------------

motifs["O single-bonded to sulfur"] = "[OX1]~S"
motifs["O single-bonded to phosphorus"] = "[OX1]~P"

motifs["O bonded to phosphorus"] = "[OX2]([*])[P]"

# ------------------------------------------------------------
# Nitrogen environments
# ------------------------------------------------------------

motifs["Primary amine N"] = "[N;H2]"
motifs["Primary amine N attached to carbon"] = "[N;H2][C]"

motifs["Secondary amine N"] = "[N;H1]"
motifs["Secondary amine N attached to one carbon"] = \
    "[N;H1]([#6X4])"

motifs["Secondary amine N attached to two carbons"] = \
    "[N;H1]([#6X4])[#6X4]"

motifs["Tertiary amine N"] = "[N;H0]"
motifs["Tertiary amine N attached to carbon"] = \
    "[N;H0]([#6X4])"

motifs["Tertiary amine N attached to two carbons"] = \
    "[N;H0]([#6X4])([#6X4])"

motifs["Tertiary amine N attached to three carbons"] = \
    "[N;H0]([#6X4])([#6X4])[#6X4]"

motifs["Cyclic N-H"] = "[N;H1;r]"
motifs["Cyclic N"] = "[N;H0;r]"

# Aromatic nitrogen
motifs["Aromatic N-H"] = "[nH]"
motifs["Aromatic N"] = "[n]"
motifs["Aromatic n-n"] = "[n]~[n]"

# Bond types
motifs["Pi bond - double"] = "[*]=[*]"
motifs["C=C"] = "[#6]=[#6]"
motifs["C=O"] = "[#6]=[#8]"
motifs["C=N"] = "[#6]=[#7]"
motifs["C=S"] = "[#6]=[#16]"
motifs["N=O"] = "[#7]=[#8]"
motifs["N=N"] = "[#7]=[#7]"

motifs["Pi bond - triple"] = "[*]#[*]"
motifs["C#C"] = "[#6]#[#6]"
motifs["C#N"] = "[#6]#[#7]"
motifs["N#N"] = "[#7]#[#7]"

motifs["Pi bond - aromatic"] = "[a]:[a]"
motifs["Aromatic C-C"] = "[c]:[c]"
motifs["Aromatic C-N"] = "[c]:[n]"
motifs["Aromatic N-N"] = "[n]:[n]"


# ============================================================
# GROVER atom features represented as SMARTS
# ============================================================

# Atomic number
# GROVER uses atomic number - 1 internally, but here we use
# the actual atomic number in the motif names.
for atomic_num in range(1, 101):
    motifs[f"Atomic number {atomic_num}"] = f"[#{atomic_num}]"

# Atomic numbers outside GROVER's supported range
motifs["Atomic number unknown"] = "[#101,#102,#103,#104,#105,#106,#107,#108,#109,#110,#111,#112,#113,#114,#115,#116,#117,#118]"

# Total degree
for degree in range(6):
    motifs[f"GROVER degree {degree}"] = f"[X{degree}]"

# Formal charge
for charge in [-1, -2, 1, 2, 0]:
    motifs[f"GROVER formal charge {charge}"] = f"[{charge:+d}]"

# Chiral tag
chiral_smarts = {
    0: "[!$([*:1][*:2])]",  # Placeholder: see note below
}


# ============================================================
# GROVER atom features with straightforward SMARTS equivalents
# ============================================================

# Total number of hydrogens
for num_h in range(5):
    motifs[f"GROVER total H {num_h}"] = f"[#1]" if num_h == -1 else f"[H{num_h}]"

# Hybridisation
motifs["GROVER hybridisation SP"] = "[X2;!$([X2]=*)]"
motifs["GROVER hybridisation SP2"] = "[X3]"
motifs["GROVER hybridisation SP3"] = "[X4]"
motifs["GROVER hybridisation SP3D"] = "[X5]"
motifs["GROVER hybridisation SP3D2"] = "[X6]"

# Aromaticity
motifs["GROVER aromatic"] = "[a]"
motifs["GROVER non-aromatic"] = "[A]"

# Hydrogen-bond donor and acceptor
motifs["GROVER hydrogen donor"] = (
    "[$([N;!H0;v3,v4&+1]),"
    "$([O,S;H1;+0]),n&H1&+0]"
)

motifs["GROVER hydrogen acceptor"] = (
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),"
    "$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=[O,N,P,S])]),"
    "n&H0&+0,"
    "$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
)

# Acidic and basic atoms
motifs["GROVER acidic"] = "[$([C,S](=[O,S,P])-[O;H1,-1])]"
motifs["GROVER basic"] = (
    "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),"
    "$([N;H1&+0]([$([C,a]);!$([C,a](=O))])"
    "[$([C,a]);!$([C,a](=O))]),"
    "$([N;H0&+0]([C;!$(C(=O))])"
    "([C;!$(C(=O))])[C;!$(C(=O))])]"
)

# Ring membership by ring size
for ring_size in range(3, 9):
    motifs[f"GROVER ring size {ring_size}"] = f"[r{ring_size}]"

filtered_motifs = {}
seen_smarts = set()

for name, smarts in motifs.items():
    if smarts not in seen_smarts:
        filtered_motifs[name] = smarts
        seen_smarts.add(smarts)


with open('dict/motif_dictionary.pkl', 'wb') as f:
    pickle.dump(filtered_motifs, f)