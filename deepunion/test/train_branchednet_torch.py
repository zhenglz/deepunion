#!/usr/bin/env python

from torch import nn
from torch import optim
import torch
import torch.utils.data
import torch.cuda
from torch.autograd import Variable
import pandas as pd
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import os, sys
from sklearn import preprocessing, externals
import pandas as pd


class BranchedNet(torch.nn.Module):

    def __init__(self, input_size_1, input_size_2):
        super().__init__()

        '''https://discuss.pytorch.org/t/how-to-train-the-network-with-multiple-branches/2152/12'''

        # channel, w, h
        self.channel = input_size_1[0]
        self.w = input_size_1[1]
        self.h = input_size_1[2]
        self.kernel = 4
        self.stride = 1
        self.padding = 0

        self.conv11 = nn.Conv2d(input_size_1[0], 128, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 128
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(128, 64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 64
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(64, 32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 32
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.relu13 = nn.ReLU(inplace=True)

        self.n_features_1 = self.channel * self.w * self.h

        # channel, w, h
        self.channel = input_size_1[0]
        self.w = input_size_2[1]
        self.h = input_size_2[2]
        self.kernel = 4
        self.stride = 1
        self.padding = 0

        self.conv21 = nn.Conv2d(input_size_2[0], 128, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 128
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.relu21 = nn.ReLU(inplace=True)

        self.conv22 = nn.Conv2d(128, 64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 64
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.relu22 = nn.ReLU(inplace=True)

        self.conv23 = nn.Conv2d(64, 32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.channel = 32
        self.w = int((self.w - self.kernel + 2 * self.padding) / self.stride + 1)
        self.h = int((self.h - self.kernel + 2 * self.padding) / self.stride + 1)
        self.relu23 = nn.ReLU(inplace=True)

        self.n_features_2 = self.channel * self.w * self.h

        total_shape = self.n_features_1 + self.n_features_2

        self.dense1 = nn.Linear(total_shape, 1000)
        self.relu4 = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(1000, 400)
        self.relu5 = nn.ReLU(inplace=True)
        self.dense3 = nn.Linear(400, 100)
        self.relu6 = nn.ReLU(inplace=True)
        self.out    = nn.Linear(100, 1)


    def forward(self, x, y):

        x = self.relu11(self.conv11(x))
        x = self.relu12(self.conv12(x))
        x = self.relu13(self.conv13(x))
        x = x.view(-1, self.n_features_1)

        y = self.relu21(self.conv21(y))
        y = self.relu22(self.conv22(y))
        y = self.relu23(self.conv23(y))
        y = y.view(-1, self.n_features_2)

        z = torch.cat((x, y), dim=1)

        z = self.relu4(self.dense1(z))
        z = self.relu5(self.dense2(z))
        z = self.relu6(self.dense3(z))

        z = self.out(z)

        return z


def rmse(output, target):
    return torch.sqrt(torch.mean((output - target) ** 2))


def pcc(output, target):
    x, y = output, target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))



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


if __name__ == "__main__":
    d = """Train or predict the features based on protein-ligand complexes.

    Examples:
    python CNN_model_keras.py -fn1 docked_training_features_12ksamples_rmsd_lessthan3a.csv 
           -fn2 training_pka_features.csv -history hist.csv -pKa_col pKa_mimic pKa -train 1

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-fn1", type=str, default="features_1.csv",
                        help="Input. The docked cplx feature set.")
    parser.add_argument("-fn2", type=str, default="features_2.csv",
                        help="Input. The PDBBind feature set.")
    parser.add_argument("-fn_y", type=str, default="features_2.csv",
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

    '''for i, fn in enumerate(args.fn1):
        if os.path.exists(fn):
            if args.index_col:
                df = pd.read_csv(fn, index_col=0, header=0).fillna(0.0)
            else:
                df = pd.read_csv(fn, header=0).fillna(0.0)

            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            print("DataFrame Shape", df.shape)
            if args.train:
                if args.pKa_col[0] in df.columns.values:
                    y = y + list(df[args.pKa_col[0]].values)
                else:
                    print("No such column %s in input file. " % args.pKa_col[0])
            if i == 0:
                X = df.values[:, :args.n_features]
            else:
                X = np.concatenate((X, df.values[:, :args.n_features]), axis=0)

            if args.pKa_col[0] in df.columns.values:
                ytrue = ytrue + list(df[args.pKa_col[0]].values)

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

            if args.train:
                yval = yval + list(df[args.pKa_col[-1]].values)

            if args.pKa_col[-1] in df.columns.values:
                # ytrue = list(ytrue) + df[args.pKa_col[1]].values

                do_eval = True'''

    X1 = pd.read_csv(args.fn1, header=0, index_col=0)
    X2 = pd.read_csv(args.fn2, header=0, index_col=0)

    y = pd.read_csv(args.fn_y, header=0, index_col=0)[args.pKa_col[0]]

    input_size_1 = (-1, 1, 9, 500)
    input_size_2 = (-1, 150)

    print("DataSet Loaded")

    if args.train > 0:

        scaler1 = preprocessing.StandardScaler()
        Xs1 = scaler1.fit_transform(X1)

        scaler2 = preprocessing.StandardScaler()
        Xs2 = scaler2.fit_transform(X2)

        if Xs1.shape[0] != Xs2.shape[0] or \
                X1.index.values != X2.index.values or \
                y.index.values != X1.index.values:
            print("The sample numbers in two branches are not equal. ")
            sys.exit(0)

        #externals.joblib.dump(scaler, args.scaler)
        print("DataSet Scaled")

        #Xtest, ytest = scaler.transform(Xval), yval
        #if len(args.reshape) == 3:
        #    Xs = Xs.reshape(-1, args.reshape[0], args.reshape[1], args.reshape[2])
        #    Xtest = Xtest.reshape(-1, args.reshape[0], args.reshape[1], args.reshape[2])
        indexer = np.arange(Xs1.shape[0])
        np.random.shuffle(indexer)

        train_part = indexer[:int(indexer.shape[0] * 0.8)]
        test_part = indexer[int(indexer.shape[0] * 0.8):]

        # Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xs, y, test_size=0.2)
        # print("Train and test split")
        X1train = Xs1[train_part].reshape(input_size_1)
        X2train = Xs2[train_part].reshape(input_size_2)
        ytrain = y.values[train_part].reshape((-1, 1))

        X1test = Xs1[test_part].reshape(input_size_1)
        X2test = Xs2[test_part].reshape(input_size_2)
        ytest = y.values[train_part].reshape((-1, 1))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #if args.method == "CNN":
        model = BranchedNet(input_size_1[1:], input_size_2[1:])
        #else:
            #model = NeuralNet(args.reshape[0])
        # if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #    model = nn.DataParallel(model)

        model = model.to(device)
        #loss_func = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9)

        print(model.eval())

        X1Train = torch.from_numpy(X1train).type(torch.FloatTensor)
        X2Train = torch.from_numpy(X2train).type(torch.FloatTensor)
        YTrain = torch.from_numpy(np.array(ytrain)).type(torch.FloatTensor)

        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(X1Train, X2Train, YTrain)
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
                inputs_1, inputs_2, labels = data
                X1, X2, Y = Variable(torch.FloatTensor(inputs_1),
                                requires_grad=False).to(device), \
                            Variable(torch.FloatTensor(inputs_2),
                                requires_grad=False).to(device), \
                            Variable(torch.FloatTensor(labels),
                                requires_grad=False).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(X1, X2)
                loss = rmse(outputs, Y)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += float(loss)

                # clear gpu memory
                del X1, X2, Y
                torch.cuda.empty_cache()
                debug_memory()

            print('[%5d] loss: %.3f' %
                  (epoch, running_loss / (i + 1)))

            '''# do validating test
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

        print('Finished Training')'''

