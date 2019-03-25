
import argparse
from argparse import RawDescriptionHelpFormatter
import sys
import os
from rdkit import Chem
from sklearn.externals import joblib
import numpy as np
import pandas as pd

from torch import nn
from torch import optim
import torch
import torch.utils.data
import torch.cuda
from torch.autograd import Variable


try:
    from PyBioMed import Pymolecule
except ImportError:
    print("Warning: PyBioMed Load Error. Could not use descriptors. ")


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 400)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(400, 200)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)

        return out


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

def SMI2Fingerprints(smi):

    m = Chem.MolFromSmiles(smi)
    f = Chem.RDKFingerprint(m).ToBitString()

    return [int(x) for x in list(f)]


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
    parser.add_argument("-reshape", type=int, default=[64, 60, 1], nargs="+",
                        help="Input. Default is 64 60 1. Reshape the dataset. ")


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load scaler and load model
    if os.path.exists(args.scaler) and os.path.exists(args.model):
        scaler = joblib.load(args.scaler)
        # pytorch model
        model = torch.load(args.model).to(device)
    else:
        print("Scaler model is not existed, exit now!")
        sys.exit(0)

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
            elif args.features == "fingerprints":
                f = SMI2Fingerprints(smi)

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

            if len(f) <= FEATURE_SIZE:
                f = f + [0.0, ] * (FEATURE_SIZE - len(f))
            else:
                f = f[:FEATURE_SIZE]

        dat = np.array(descriptors)

        # do the prediction now
        #try:
        Xpred = scaler.transform(dat)
        if len(args.reshape) == 3:
            Xpred = Xpred.reshape((-1, args.reshape[0], args.reshape[1], args.reshape[2]))
        else:
            Xpred = Xpred.reshape((-1, args.reshape[0]))

        Xpred = torch.from_numpy(Xpred).type(torch.FloatTensor).to(device)
        ypred = list(model(Xpred).cpu().detach().numpy().ravel())
            
        #except RuntimeError:
        #    ypred = [99., ] * CHUNK

        pred_logS += ypred

        del ypred
        torch.cuda.empty_cache()

        if args.v:
            print("PROGRESS: %12d out of %20d."%(i*CHUNK, df.shape[0]))

        output = pd.DataFrame()
        output['ID'] = df.ID.values[: len(success)]
        output['SMI'] = df.SMI.values[: len(success)]
        output['logS_pred'] = pred_logS
        output['success'] = success

        output.to_csv(args.out, header=True, index=True, float_format="%.3f", )

