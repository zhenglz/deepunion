import sys
from rdkit import Chem
import pandas as pd
from PyBioMed import Pymolecule


def SMI2Descriptor(smi):

    #m = Chem.MolFromSmiles(smi)
    mol = Pymolecule.PyMolecule()
    mol.ReadMolFromSmile(smi)

    alldes = mol.GetAllDescriptor()

    keys = sorted(alldes.keys())
    features = [alldes[x] for x in keys]

    return features, keys


if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1], header=0, sep=",")
    df = df.dropna()

    SMIs = df.SMILES.values

    descriptors = []
    names = []
    count = 0
    keys = []
    smiles = []
    for (smi, n) in zip(SMIs, df["Compound ID"].values):
        count += 1
        if count > -2 :
            try:
                d, k = SMI2Descriptor(smi)
                keys = k
                smiles.append(smi)
                names.append(n)
                descriptors.append(d)
            except:
                print("Failed %s " % n)


            print("Progress: %d " % count)

            if count % 20 == 0:
                dat = pd.DataFrame(descriptors, columns=keys)
                dat['Name'] = names
                dat['SMI'] = smiles

                dat.to_csv("descriptors_%s" % sys.argv[1], header=True, index=False,
                           sep=",", float_format="%.3f")

