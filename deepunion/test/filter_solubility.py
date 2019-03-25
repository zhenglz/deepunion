
import argparse
from argparse import RawDescriptionHelpFormatter
import sys
import os
from rdkit import Chem
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from PyBioMed import Pymolecule


def SMI2Descriptor(smi):
    """
    Extract molecular descriptors from SMILES code with PyBioMed
    :param smi:
    :return:
    """
    #m = Chem.MolFromSmiles(smi)
    try:
        mol = Pymolecule.PyMolecule()
        mol.ReadMolFromSmile(smi)

        alldes = mol.GetAllDescriptor()

        keys = sorted(alldes.keys())
        features = [alldes[x] for x in keys]
    except RuntimeError:
        features, keys = None, None

    return features, keys


def SMILE2Strings(smi, length=100):
    if len(smi) <= length:
        return list(smi) + (100-len(smi)) * ['X', ]
    else:
        return list(smi)[:100]

def Fingerprints(smi):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="",
                                     formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-smi", type=str, default="SMILES.list",
                        help="Input, optional. \n"
                             "The input smiles code file. Two columns should present in \n"
                             "the file, the first columns is the SMILES code, the 2nd is"
                             "the unique ID of the molecule. ")
    parser.add_argument("-out", type=str, default="Output_solubility.list",
                        help="Output, optional. \n")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Input, optional. \n"
                             "The scaler for the standardization of the descriptors. ")
    parser.add_argument("-model", type=str, default="RFRegression.model",
                        help="Input, optional. \n"
                             "The model for the regression. ")
    parser.add_argument("-v", type=int, default=1,
                        help="Input, optional. Default is 1. \n"
                             "Whether output detail information. ")
    parser.add_argument("-features", type=str, default="onehot",
                        help="Input, optional. Default is descriptors. "
                             "Choices: descriptors, onehot, fingerprints. \n")
    parser.add_argument("-chunk", default=100, type=int,
                        help="Input, optional. ")
    parser.add_argument("-feature_size", default=100, type=int,
                        help="Input, optional. ")

    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()

    CHUNK = args.chunk
    pred_logS = []
    FEATURE_SIZE = args.feature_size

    # load the smile code
    df = pd.read_csv(args.smi, header=-1, sep=" ")
    df.columns = ['SMI', 'ID']
    df = df.dropna()
    SMILES = df.SMI.values

    # number of chunks
    SIZE = int(df.shape[0] / CHUNK)
    success = []

    # predict logS per chunk
    for i in range(SIZE):
        descriptors = []

        smiles = SMILES[CHUNK*i: CHUNK*i+CHUNK]
        if i == SIZE - 1:
            smiles = SMILES[CHUNK * i:]

        for smi in smiles:
            # get descriptors from SMILES
            if args.features == "descriptors":
                f, k = SMI2Descriptor(smi)
            elif args.features == "onehot":
                f = SMILE2Strings(smi, 100)

            if f is None:
                f = [0.0, ] * FEATURE_SIZE
                success.append(0)
                descriptors.append(f)
            elif len(f) == FEATURE_SIZE:
                success.append(1)
                descriptors.append(f)
            else:
                descriptors.append([0.0, ] * FEATURE_SIZE)
                success.append(0)

        dat = np.array(descriptors)

        # load scaler and load model
        if os.path.exists(args.scaler) and os.path.exists(args.model) :
            scaler = joblib.load(args.scaler)
            model  = joblib.load(args.model)
        else:
            print("Scaler model is not existed, exit now!")
            sys.exit(0)

        # do the prediction now
        try:
            Xpred = scaler.transform(dat)
            ypred = list(model.predict(Xpred).ravel())
            
        except RuntimeError:
            ypred = [99., ] * FEATURE_SIZE

        pred_logS += ypred

        if args.v:
            print("PROGRESS: %12d out of %20d."%(i*CHUNK, df.shape[0]))

        output = pd.DataFrame()
        output['ID'] = df.ID.values[: len(success)]
        output['SMI'] = df.SMI.values[: len(success)]
        output['logS_pred'] = pred_logS
        output['success'] = success

        output.to_csv(args.out, header=True, index=True, float_format="%.3f", )

