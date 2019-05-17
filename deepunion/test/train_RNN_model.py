import sys
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
import pandas as pd
import sys
from torch.autograd import Variable


def rmse(output, target):

    return torch.sqrt(torch.mean((output - target) ** 2))


def PCC(output, target):


    vx = target.view(-1) - torch.mean(target.view(-1))
    vy = output.view(-1) - torch.mean(output.view(-1))

    P = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    #P = vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))

    #x = output.detach().numpy().ravel()
    #y = target.detach().numpy().ravel()

    #return stats.pearsonr(x, y)[0]
    return P


def PCC_loss(output, target):


    vx = target.view(-1) - torch.mean(target.view(-1))
    vy = output.view(-1) - torch.mean(output.view(-1))

    P = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    #P = vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))

    #x = output.detach().numpy().ravel()
    #y = target.detach().numpy().ravel()

    #return stats.pearsonr(x, y)[0]
    return 1 - P


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
    RMSE = torch.sqrt(torch.mean((output - target) ** 2))

    vx = target - torch.mean(target)
    vy = output - torch.mean(output)

    #pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    PCC = vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))

    return alpha * RMSE + (1-alpha) * (1- PCC)


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))



def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


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


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=1, downsample=None):

        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)

        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8,ceil_mode=False) #  nn.AvgPool2d需要添加参数ceil_mode=False，否则该模块无法导出为onnx格式
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # 残差直接映射部分
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    d = """Train or predict the features based on protein-ligand complexes.

    Examples:
    python CNN_model_keras.py -fn1 docked_training_features_12ksamples_rmsd_lessthan3a.csv 
           -fn2 training_pka_features.csv -history hist.csv -pKa_col pKa_mimic pKa -train 1

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-fn_train", type=str, default=["features_1.csv", ], nargs="+",
                        help="Input. The docked cplx feature training set.")
    parser.add_argument("-fn_validate", type=str, default=["features_2.csv", ], nargs="+",
                        help="Input. The PDBBind feature validating set.")
    parser.add_argument("-fn_test", type=str, default=["features_2.csv", ], nargs="+",
                        help="Input. The PDBBind feature testing set.")
    parser.add_argument("-y_col", type=str, nargs="+", default=["pKa_relu", "pKa_true"],
                        help="Input. The pKa colname as the target. ")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
    parser.add_argument("-model", type=str, default="DNN_Model.h5",
                        help="Output. The trained DNN model file to save. ")
    parser.add_argument("-log", type=str, default="",
                        help="Output. The logger file name to save. ")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa values file name to save. ")
    parser.add_argument("-lr_init", type=float, default=0.001,
                        help="Input. Default is 0.001. The initial learning rate. ")
    parser.add_argument("-epochs", type=int, default=100,
                        help="Input. Default is 100. The number of epochs to train. ")
    parser.add_argument("-batch", type=int, default=128,
                        help="Input. Default is 128. The batch size. ")
    parser.add_argument("-patience", type=int, default=20,
                        help="Input. Default is 20. The patience steps. ")
    parser.add_argument("-delta_loss", type=float, default=0.01,
                        help="Input. Default is 0.01. The delta loss for early stopping. ")
    parser.add_argument("-dropout", type=float, default=0.1,
                        help="Input. Default is 0.1. The dropout rate. ")
    parser.add_argument("-alpha", type=float, default=0.1,
                        help="Input. Default is 0.1. The alpha value. ")
    parser.add_argument("-train", type=int, default=1,
                        help="Input. Default is 1. Whether train or predict. \n"
                             "1: train, 0: predict. ")
    parser.add_argument("-pooling", type=int, default=0,
                        help="Input. Default is 0. Whether using maxpooling. \n"
                             "1: with pooling, 0: no pooling. ")
    parser.add_argument("-n_features", default=3840, type=int,
                        help="Input. Default is 3840. Number of features in the input dataset.")
    parser.add_argument("-reshape", type=int, default=[64, 60, 1], nargs="+",
                        help="Input. Default is 64 60 1. Reshape the dataset. ")
    parser.add_argument("-remove_H", type=int, default=0,
                        help="Input, optional. Default is 0. Whether remove hydrogens. ")
    parser.add_argument("-hidden_layers", type=int, default=[400, 200, 100], nargs="+",
                        help="Input, optional. Default is 400 200 100. The hidden layer units.")
    parser.add_argument("-method", type=str, default='CNN',
                        help="Input, optional. Default is CNN. Options: CNN, DNN. The learning network type.")

    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)

    X, y = None, []
    do_eval = False

    global alpha
    alpha = args.alpha

    for i, fn in enumerate(args.fn_train):
        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            print("DataFrame Shape", df.shape)

            if args.train:
                if args.y_col[0] in df.columns.values:
                    y = y + list(df[args.y_col[0]].values)
                else:
                    print("No such column %s in input file. " % args.y_col[0])

            if i == 0:
                X = df.values[:, :args.n_features]
            else:
                X = np.concatenate((X, df.values[:, :args.n_features]), axis=0)

    Xval, yval = None, []
    for i, fn in enumerate(args.fn_validate):
        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            if i == 0:
                Xval = df.values[:, :args.n_features]
            else:
                Xval = np.concatenate((Xval, df.values[:, :args.n_features]), axis=0)

            if args.train:
                yval = yval + list(df[args.y_col[-1]].values)

    Xtest, ytest = None, []
    for i, fn in enumerate(args.fn_test):
        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            if i == 0:
                Xtest = df.values[:, :args.n_features]
            else:
                Xtest = np.concatenate((Xtest, df.values[:, :args.n_features]), axis=0)

            if args.train:
                ytest = ytest + list(df[args.y_col[-1]].values)

    print("DataSet Loaded")

    if args.train > 0:

        scaler = preprocessing.StandardScaler()
        X_train_val = np.concatenate((X, Xval), axis=0)
        scaler.fit(X_train_val)

        joblib.dump(scaler, args.scaler)

        if args.method == "CNN":
            Xtrain = scaler.transform(X).reshape((-1, args.reshape[0],
                                                  args.reshape[1],
                                                  args.reshape[2]))
            Xval = scaler.transform(Xval).reshape((-1, args.reshape[0],
                                                   args.reshape[1],
                                                   args.reshape[2]))
            Xtest = scaler.transform(Xtest).reshape((-1, args.reshape[0],
                                                     args.reshape[1],
                                                     args.reshape[2]))
        else:
            Xtrain = scaler.transform(X)
            Xval = scaler.transform(Xval)
            Xtest = scaler.transform(Xtest)

        ytrain = np.array(y).reshape((-1, 1))
        yval = np.array(yval).reshape((-1, 1))
        ytest = np.array(ytest).reshape((-1, 1))

        print("DataSet Scaled")

        torch.cuda.set_device(1) # select a gpu id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model defined
        model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

        loss_func = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9)

        # print model summary
        print(model.eval())

        XTrain = torch.from_numpy(Xtrain).type(torch.FloatTensor)
        YTrain = torch.from_numpy(ytrain).type(torch.FloatTensor)

        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(XTrain, YTrain)
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=False)

        min_val = [[0, 999.9], ]
        delta = args.delta
        patience = args.patience

        history = []

        for epoch in range(args.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_pcc = 0.0
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
                #loss = PCC_loss(outputs, Y)
                loss = rmse(outputs, Y)
                #loss = loss_func(outputs, Y)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += float(loss)

                r = PCC(outputs, Y)
                running_pcc = float(r)
                # clear gpu memory
                del X, Y, r
                torch.cuda.empty_cache()
                #debug_memory()

                # do validating test
            Xval = Variable(torch.from_numpy(Xtest).type(torch.FloatTensor),
                            requires_grad=False).to(device)
            # Xval = torch.Tensor.cpu(Xval)
            yval = model(Xval).cpu()

            val_rmse = float(rmse(yval, torch.from_numpy(np.array(ytest)).type(torch.FloatTensor)))
            val_pcc = float(PCC(yval, torch.from_numpy(ytest.ravel()).type(torch.FloatTensor)))

            del Xval, yval
            debug_memory()
            torch.cuda.empty_cache()

            print('[%5d] loss: %.3f, pcc: %.3f val_loss: %.3f, val_pcc: %.3f' %
                  (epoch, running_loss / (i + 1), running_pcc, val_rmse, val_pcc))

            history.append([epoch, running_loss / (i + 1), running_pcc, val_rmse, val_pcc])

            # early stopping
            # do validating test
            if min_val[-1][1] - val_rmse >= delta:
                print("Model improve from %.3f to %.3f . Save model to %s " % (min_val[-1][1], val_rmse, args.model))
                torch.save(model.state_dict(), args.model)
                min_val.append([epoch, val_rmse])

            else:
                if epoch - min_val[-1][0] >= patience:
                    print("Get best model at epoch = %d" % min_val[-1][0])
                    break
                else:
                    pass

            hist = pd.DataFrame(history, columns=['epochs', 'loss', 'pcc', 'val_loss', 'val_pcc'])
            hist.to_csv(args.log, header=True, index=False, float_format="%.4f")

        print('Finished Training')
