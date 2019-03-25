import mdtraj as mt
from rdkit import Chem
import sys


def long2short(code):

    with open("amino-acid.lib") as lines:
        code_mapper = {}
        for s in lines:
            if len(s) and s[0] != "#":
                code_mapper[s.split()[2]] = s.split()[3]

    if code in code_mapper.keys():
        return code_mapper[code]
    else:
        return ""


def res_seq(pdb):

    p = mt.load_pdb(pdb)
    seq = list(p.topology.residues)
    seq = [str(x)[:3] for x in seq if len(str(x)) >= 3]

    return list(map(long2short, seq))


def lig_smiles(pdb):

    m = Chem.MolFromMol2File(pdb)
    return Chem.MolToSmiles(m)


if __name__ == "__main__":

    p = sys.argv[1]
    o = sys.argv[2]
    s = sys.argv[4]
    lig = sys.argv[3]

    seq = res_seq(p)

    with open(o, "w") as tofile:
        l = p +","+ "".join(seq) + "\n"
        tofile.write(l)

    with open(s, "w") as tofile:
        l = lig+","+lig_smiles(lig)
        tofile.write(l)

