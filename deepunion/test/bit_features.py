from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
import sys
import pandas as pd
import numpy as np


def fingerprints(smi):

    mol = Chem.MolFromSmiles(smi)

    f = FingerprintMols.FingerprintMol(mol)

    bits = f.ToBitString()

    return [int(x) for x in list(bits)]


if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1], header=0, sep=",")
    out = sys.argv[2]

    bits_list = []
    smiles = df.smiles.values
    for i, smi in enumerate(smiles):
        if i % 100 == 0:
            print(i)
        bits_list.append(fingerprints(smi))

    bits = pd.DataFrame(bits_list, columns=["FP"+str(x) for x in np.arange(len(bits_list[0]))])
    bits.index = df.values[:, 0]
    bits['logD'] = df['exp'].values

    bits.to_csv(out, header=True, index=True, float_format="%.3f")

