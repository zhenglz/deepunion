#!/usr/bin/env python

from sklearn import preprocessing
import numpy as np
from sklearn.externals import joblib
import argparse
from argparse import RawTextHelpFormatter
import os
from scipy import stats
from torch import nn
from torch import optim
import torch
import torch.utils.data
import torch.cuda
from torch.autograd import Variable
import pandas as pd
import sys
from sklearn import model_selection


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


class SimpleCNN(nn.Module):

    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()

        # channel, w, h
        self.channel = input_size[0]
        self.w = input_size[1]
        self.h = input_size[2]
        self.kernel = 4
        self.stride = 1
        self.padding = 0

        self.conv1 = nn.Conv2d(1, 128, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 128
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 64
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.bn2 = nn.BatchNorm2d(self.channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 32
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.bn3 = nn.BatchNorm2d(self.channel)
        self.relu3 = nn.ReLU(inplace=True)

        self.dense1 = nn.Linear(self.channel * self.w * self.h, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.relu4 = nn.ReLU(inplace=True)

        self.dense2 = nn.Linear(1000, 400)
        self.bn5 = nn.BatchNorm1d(400)
        self.relu5 = nn.ReLU(inplace=True)

        self.dense3 = nn.Linear(400, 100)
        self.bn6 = nn.BatchNorm1d(100)
        self.relu6 = nn.ReLU(inplace=True)

        self.out = nn.Linear(100, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = x.view(-1, self.channel * self.w * self.h)

        x = self.relu4(self.bn4(self.dense1(x)))
        x = self.relu5(self.bn5(self.dense2(x)))
        x = self.relu6(self.bn6(self.dense3(x)))

        x = self.out(x)

        return x


def rmse(output, target):
    return torch.sqrt(torch.mean((output - target) ** 2))


def pcc(output, target):
    x, y = output, target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def PCC(output, target):
    x = output.detach().numpy().ravel()
    y = target.detach().numpy().ravel()

    return stats.pearsonr(x, y)[0]


def remove_shell_features(dat, shell_index, features_n=64):
    df = dat.copy()

    start = shell_index * features_n
    end = start + features_n

    zeroes = np.zeros((df.shape[0], features_n))

    df[:, start:end] = zeroes

    return df


def remove_atomtype_features(dat, feature_index, shells_n=60):
    df = dat.copy()

    for i in range(shells_n):
        ndx = i * 64 + feature_index

        zeroes = np.zeros(df.shape[0])
        df[:, ndx] = zeroes

    return df


def remove_all_hydrogens(dat, n_features):
    df = dat.copy()

    for f in df.columns.values[:n_features]:
        if "H_" in f or "_H_" in f:
            v = np.zeros(df.shape[0])
            df[f] = v

    return df


def rmse_pcc_loss(output, target):
    alpha = 0.8
    rmse = torch.sqrt(torch.mean((output - target) ** 2))

    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    return alpha * rmse + (1 - alpha) * (1 - pcc)


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    # for line in sorted(tensors.items()):
    #    print('{}\t{}'.format(*line))
    # print("Clean items number ", len(tensors.items()))


if __name__ == "__main__":
    d = """Train or predict the features based on protein-ligand complexes.

    The input scaler is a scikit-learn standardscaler object.
    The model is saved from a DNN model built with pytorch.

    Examples:
    python CNN_model_keras.py -fn1 docked_training_features_12ksamples_rmsd_lessthan3a.csv 
           -fn2 training_pka_features.csv -history hist.csv -pKa_col pKa_mimic pKa -train 1

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-fn1", type=str, default=["features_1.csv", ], nargs="+",
                        help="Input. The docked cplx feature set.")
    parser.add_argument("-fn2", type=str, default=["features_2.csv", ], nargs="+",
                        help="Input. The PDBBind feature set.")
    parser.add_argument("-history", type=str, default="history.csv",
                        help="Output. The history information. ")
    parser.add_argument("-pKa_col", type=str, nargs="+", default=["pKa_relu", "pKa_true"],
                        help="Input. The pKa colname as the target. ")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
    parser.add_argument("-model", type=str, default="DNN_Model.h5",
                        help="Output. The trained DNN model file to save. ")
    parser.add_argument("-log", type=str, default="logger.csv",
                        help="Output. The logger file name to save. ")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa values file name to save. ")
    parser.add_argument("-lr_init", type=float, default=0.001,
                        help="Input. Default is 0.001. The initial learning rate. ")
    parser.add_argument("-epochs", type=int, default=100,
                        help="Input. Default is 100. The number of epochs to train. ")
    parser.add_argument("-train", type=int, default=1,
                        help="Input. Default is 1. Whether train or predict. \n"
                             "1: train, 0: predict. ")
    parser.add_argument("-n_features", default=3840, type=int,
                        help="Input. Default is 3840. Number of features in the input dataset.")
    parser.add_argument("-reshape", type=int, default=[64, 60, 1], nargs="+",
                        help="Input. Default is 64 60 1. Reshape the dataset. ")
    parser.add_argument("-remove_H", type=int, default=0,
                        help="Input, optional. Default is 0. Whether remove hydrogens. ")
    parser.add_argument("-index_col", type=int, default=1,
                        help="Input. Default is 1. Whether include the index col. ")
    parser.add_argument("-method", default="CNN", type=str,
                        help="Input, optional. Default is CNN. "
                             "Which DNN models to use. Options: CNN, DNN.")

    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)

    X, y = None, []
    Xval, yval = None, []
    do_eval = False
    ytrue = []

    for i, fn in enumerate(args.fn1):
        if os.path.exists(fn):
            if args.index_col:
                df = pd.read_csv(fn, index_col=0, header=0).fillna(0.0)
            else:
                df = pd.read_csv(fn, header=0).fillna(0.0)

            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            if i == 0:
                X = df.values[:, :args.n_features]
            else:
                X = np.concatenate((X, df.values[:, :args.n_features]), axis=0)

            print("DataFrame Shape", df.shape)

            if args.train > 0 and args.pKa_col[0] in df.columns.values:
                y = y + list(df[args.pKa_col[0]].values)

                do_eval = True

    for i, fn in enumerate(args.fn2):
        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            if i == 0:
                Xval = df.values[:, :args.n_features]
            else:
                Xval = np.concatenate((X, df.values[:, :args.n_features]), axis=0)

            if args.train and args.pKa_col[-1] in df.columns.values:
                yval = yval + list(df[args.pKa_col[-1]].values)
                do_eval = True

    print("DataSet Loaded")

    if args.train > 0:

        scaler = preprocessing.StandardScaler()
        Xs = scaler.fit_transform(X)
        joblib.dump(scaler, args.scaler)
        print("DataSet Scaled")

        Xtest, ytest = scaler.transform(Xval), yval
        if len(args.reshape) == 3:
            Xs = Xs.reshape(-1, args.reshape[0], args.reshape[1], args.reshape[2])
            Xtest = Xtest.reshape(-1, args.reshape[0], args.reshape[1], args.reshape[2])

        # Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xs, y, test_size=0.2)
        # print("Train and test split")
        Xtrain, ytrain = Xs, y

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.method == "CNN":
            model = SimpleCNN(args.reshape)
        else:
            model = NeuralNet(args.reshape[0])
        # if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #    model = nn.DataParallel(model)

        model = model.to(device)
        loss_func = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9)

        print(model.eval())

        XTrain = torch.from_numpy(Xtrain).type(torch.FloatTensor)
        YTrain = torch.from_numpy(np.array(ytrain).reshape((-1, 1))).type(torch.FloatTensor)

        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(XTrain, YTrain)
        # test = torch.utils.data.TensorDataset(XTest, YTest)

        train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False)

        min_val = [[0.0, 999.9], ]
        delta = 0.0001
        patience = 40
        history = []

        for epoch in range(args.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs
                inputs, labels = data
                X, Y = Variable(torch.FloatTensor(inputs),
                                requires_grad=False).to(device), \
                       Variable(torch.FloatTensor(labels),
                                requires_grad=False).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(X)
                loss = rmse(outputs, Y)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += float(loss)

                # clear gpu memory
                del X
                del Y
                torch.cuda.empty_cache()
                debug_memory()

            # do validating test
            Xv = Variable(torch.from_numpy(Xtest).type(torch.FloatTensor),
                          requires_grad=False).to(device)
            # Xval = torch.Tensor.cpu(Xval)
            yv = model(Xv).cpu()

            val_rmse = float(rmse(yv, torch.from_numpy(np.array(ytest).reshape(-1, 1)).type(torch.FloatTensor)))
            val_pcc = float(PCC(yv, torch.from_numpy(np.array(ytest).reshape(-1, 1)).type(torch.FloatTensor)))

            # del Xval, yval
            # debug_memory()
            # torch.cuda.empty_cache()

            print('[%5d] loss: %.3f, val_loss: %.3f, val_pcc: %.3f, val_r: %.3f' %
                  (epoch, running_loss / (i + 1), val_rmse, val_pcc,
                   float(pcc(yv, torch.from_numpy(np.array(ytest).reshape(-1, 1)).type(torch.FloatTensor)))))

            del Xv, yv
            debug_memory()
            torch.cuda.empty_cache()

            # early stopping
            # do validating test
            history.append([epoch, running_loss / (i + 1), val_rmse, val_pcc])
            log = pd.DataFrame(history)
            log.columns = ['epochs', 'loss', 'val_loss', 'val_pcc']
            log.to_csv(args.log, header=True, index=False, float_format="%.4f")

            if min_val[-1][1] - val_rmse >= delta:
                print("Model improve from %.3f to %.3f . Save model to %s " % (min_val[-1][1], val_rmse, args.model))
                torch.save(model, args.model)
                min_val.append([epoch, val_rmse])
            else:
                if epoch - min_val[-1][0] >= patience:
                    print("Get best model at epoch = %d" % min_val[-1][0])
                    break
                else:
                    pass

        print('Finished Training')

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        scaler = joblib.load(args.scaler)
        model = torch.load(args.model)

        # if len(args.reshape) == 3:
        Xs = scaler.transform(X).reshape(tuple([-1, ] + args.reshape))

        Xpred = Variable(torch.from_numpy(Xs).type(torch.FloatTensor),
                         requires_grad=False).to(device)

        if "gpu" in device:
            ypred = model(Xpred).cpu()
        else:
            ypred = model(Xpred)

        df = pd.DataFrame()

        if do_eval:
            r = PCC(ypred.numpy().ravel(), np.array(y))
            error = float(rmse(ypred, torch.from_numpy(np.array(y).reshape(-1, 1)).type(torch.FloatTensor)))
            print("PCC: %.3f \nRMSE: %.3f" % (r, error))
            df['real'] = np.array(y)

        df['predicted'] = ypred.numpy().ravel()
        # df.index = X.index
        df.to_csv(args.out, header=True, index=True, float_format="%.3f")
