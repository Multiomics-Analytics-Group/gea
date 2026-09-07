from rdkit.Chem import FunctionalGroups
import pickle
from pathlib import Path

directory = Path("dict")
directory.mkdir(parents=True, exist_ok=True)

motifs = {}
fg_tree = FunctionalGroups.BuildFuncGroupHierarchy()

for fg in fg_tree:
    motifs[fg.name] = fg.smarts
    subfg = fg.children
    for child in subfg:
        motifs[fg.name + " - " + child.name] = child.smarts

motifs['Amine - Cyclic'] = "[N;R;$(N-[#6]);!$(N=*);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]"

# General carbon environments
motifs["C-H3"] = "[CH3]"
motifs["C-H2"] = "[CH2]"
motifs["C-H1"] = "[CH]"
motifs["C-H0"] = "[C;H0]"

motifs["aromatic carbon"] = "[c]"
motifs["aliphatic carbon"] = "[C;A]"
motifs["cyclic carbon"] = "[C;R]"
motifs["acyclic carbon"] = "[C;!R]"

motifs["aromatic cyclic carbon"] = "[c;R]"
motifs["aliphatic cyclic carbon"] = "[C;A;R]"
motifs["aliphatic acyclic carbon"] = "[C;A;!R]"

motifs["aromatic CH"] = "[cH]"
motifs["aromatic C substituted"] = "[cH0]"

motifs['aromatic CH adjacent to substituted carbon'] = '[cH](:[cH0])'
motifs['aromatic carbon attached to aliphatic carbon'] = '[c]-[C;!a]'
motifs['aromatic carbon attached to heteroatom'] = '[c]-[!#6;!#1]'

# Hybridization
motifs["C sp3"] = "[C;X4]"
motifs["C sp2"] = "[C;X3]"
motifs["C sp"] = "[C;X2]"

# Alkane-specific
motifs["alkane"] = "[C;X4;!a]"
motifs["alkane CH3"] = "[CH3;X4]"
motifs["alkane CH2"] = "[CH2;X4]"
motifs["alkane CH"] = "[CH;X4]"
motifs["alkane C"] = "[C;H0;X4]"

# Aromatic
motifs["benzene ring"] = "c1ccccc1"

# Amides
motifs["amide"] = "[CX3](=O)[NX3]"
motifs["Amide - Primary"] = "[CX3](=O)[NH2]"
motifs["Amide - Secondary"] = '[CX3](=O)[NH1]([#6])'
motifs["Amide - Tertiary"] = '[CX3](=O)[N;H0]([#6])[#6]'
motifs["Lactam"] = '[C;R](=O)[N;R]'
motifs["Amide - Aromatic"] = '[c][CX3](=O)[NX3]'
motifs["Amide - N-Aromatic"] = '[CX3](=O)[N;X3]-c'

# alkene
motifs["alkene"] = '[C;X3;!a]=[C;X3;!a]'
motifs["alkene - terminal"] = '[C;X3;!a]=[CH2]'
motifs["alkene - internal"] = '[C;X3;!a]([#6])=[C;X3;!a]([#6])'
motifs["alkene - conjugated"] = '[C;X3;!a]=[C;X3;!a]-[C;X3;!a]=[C;X3;!a]'

# Alkynes
motifs["alkyne"] = "[CX2]#[CX2]"
motifs["alkyne - terminal"] = '[C;X2]#[CH]'
motifs["alkyne - internal"] = '[C;X2]([#6])#[C;X2]([#6])'

# Ether
motifs["ether"] = '[OD2]([#6])[#6]'
motifs["ether - aliphatic"] = '[O;D2]([C;!a])[C;!a]'
motifs["ether - aromatic"] = '[O;D2]([c])[#6]'
motifs["ether - diaryl"] = '[O;D2](c)c'
motifs["ether - cyclic"] = '[O;R;D2]'
motifs["epoxide"] = "[OX2r3]([#6r3])[#6r3]"
motifs["ether - 5-membered ring"] = '[O;R;r5]'
motifs["ether - 6-membered ring"] = '[O;R;r6]'

# Thiol
motifs["thiol"] = "[SX2H]"
motifs["thiol - aliphatic"] = '[S;H1;$(S-[C;!a])]'
motifs["thiol - aromatic"] = '[S;H1;$(S-c)]'

# Ketone
motifs["ketone"] = "[#6][CX3](=O)[#6]"
motifs["ketone - aliphatic"] = '[C;!a][CX3](=O)[C;!a]'
motifs["ketone - aromatic"] = '[c][CX3](=O)[#6]'
motifs["ketone - cyclic"] = '[C;R][CX3;R](=O)[C;R]'
motifs["ketone - alpha,beta-unsaturated"] = '[C;X3;!a]=[C;X3;!a][CX3](=O)[#6]'

# Ester
motifs["ester"] = "[CX3](=O)[OX2H0][#6]"
motifs["ester - aliphatic"] = '[C;!a][CX3](=O)[OX2H0][C;!a]'
motifs["ester - aromatic carbonyl"] = '[c][CX3](=O)[OX2H0][#6]'
motifs["ester - aryl oxygen"] = '[CX3](=O)[OX2H0][c]'
motifs["lactone"] = '[C;R](=O)[O;R]'

# Nitrile
motifs["nitrile"] = "[CX2]#[NX1]"
motifs["nitrile - aliphatic"] = '[C;X2;!a]#[N;X1]'
motifs["nitrile - aromatic"] = '[c]-[C;X2]#[N;X1]'
motifs["nitrile - cyclic"] = '[C;X2;R]#[N;X1]'

# Imine
motifs["imine"] = "[CX3]=[NX2]"
motifs["cyclic imine"] = "[C;R]=[N;R]"
motifs["aldimine"] = "[CH;X3]=[NX2]"
motifs["ketimine"] = "[C;X3;H0]=[NX2]"


# Anhydride
motifs["anhydride"] = "[CX3](=O)[OX2][CX3](=O)"
motifs["cyclic anhydride"] = "[C;R](=O)[O;R][C;R](=O)"

# Sulfide
motifs["sulfide"] = "[SX2]([#6])[#6]"
motifs["sulfide - aliphatic"] = "[S;X2]([C;!a])[C;!a]"
motifs["sulfide - aromatic"] = "[S;X2](c)[#6]"
motifs["sulfide - diaryl"] = "[S;X2](c)c"
motifs["sulfide - cyclic"] = "[S;X2;R]"

# Sulfonamide
motifs["sulfonamide"] = '[S](=O)(=O)([#6])[NH]'
motifs["sulfonamide - primary"] = '[S](=O)(=O)([#6])[NH2]'
motifs["sulfonamide - secondary"] = '[S](=O)(=O)([#6])[NH1]([#6])'
motifs["sulfonamide - tertiary"] = '[S](=O)(=O)([#6])[N;H0]([#6])[#6]'

# Extra halogens
motifs['Halogen - Fluorine'] = '[F]'
motifs['Halogen - Chlorine'] = '[Cl]'
motifs['Halogen - Iodine'] = '[I]'
motifs['Fluorine - Aromatic'] = '[F;$(*-!@c)]'
motifs['Fluorine - Aliphatic'] = '[F;$(*-!@C)]'

# General annotations
motifs["aromatic atom"] = "[a]"
motifs["aliphatic atom"] = "[A]"
motifs["cyclic atom"] = "[R]"
motifs["acyclic atom"] = "[!R]"

motifs["aromatic cyclic atom"] = "[a;R]"
motifs["aliphatic cyclic atom"] = "[A;R]"
motifs["aliphatic acyclic atom"] = "[A;!R]"

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



with open('dict/motif_dictionary.pkl', 'wb') as f:
    pickle.dump(motifs, f)