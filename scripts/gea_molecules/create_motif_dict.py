from rdkit.Chem import FunctionalGroups
import pickle

motifs = {}
fg_tree = FunctionalGroups.BuildFuncGroupHierarchy()

for fg in fg_tree:
    subfg = fg.children
    for child in subfg:
        motifs[fg.name + " - " + child.name] = child.smarts

motifs["alkane"] = "[CX4]"
motifs["alkene"] = "[CX3]=[CX3]"
motifs["alkyne"] = "[CX2]#[CX2]"
motifs["ether"] = "[OD2]([#6])[#6]"
motifs["benzene ring"] = "c1ccccc1"
motifs["thiol"] = "[SX2H]"
motifs["ketone"] = "[#6][CX3](=O)[#6]"
motifs["ester"] = "[CX3](=O)[OX2H0][#6]"
motifs["amide"] = "[CX3](=O)[NX3]"
motifs["nitrile"] = "[CX2]#[NX1]"
motifs["epoxide"] = "[OX2r3]([#6r3])[#6r3]"
motifs["imine"] = "[CX3]=[NX2]"
motifs["anhydride"] = "[CX3](=O)[OX2][CX3](=O)"
motifs["sulfide"] = "[SX2]([#6])[#6]"

with open('dict/motif_dictionary.pkl', 'wb') as f:
    pickle.dump(motifs, f)