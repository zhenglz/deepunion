import sys
from sklearn import preprocessing, model_selection
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import argparse
from argparse import RawTextHelpFormatter
import os
from scipy import stats


def rmse(y_true, y_pred):

    dev = np.square(y_true.ravel() - y_pred.ravel())

    return np.sqrt(np.sum(dev) / y_true.shape[0])


def pcc(y_true, y_pred):
    pcc = stats.pearsonr(y_true, y_pred)
    return pcc[0]


def PCC_RMSE(y_true, y_pred):

    alpha = 0.7

    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

    pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)

    pcc = tf.where(tf.is_nan(pcc), 0.25, pcc)

    return alpha * pcc + (1-alpha) * rmse


def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):

    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


def create_model(input_size, lr=0.0001):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(128, 4, 1, input_shape=input_size))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(64, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(32, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(40, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))

    sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6, )
    model.compile(optimizer=sgd, loss=PCC_RMSE, metrics=["mse", PCC, RMSE])

    return model


if __name__ == "__main__":
    d = """Train or predict the features based on protein-ligand complexes.
    
    Examples:
    python CNN_model_keras.py -fn1 docked_training_features_12ksamples_rmsd_lessthan3a.csv 
           -fn2 training_pka_features.csv -history hist.csv -pKa_col pKa_mimic pKa -train 1
           
    """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-fn1", type=str, default="features_1.csv",
                        help="Input. The docked cplx feature set.")
    parser.add_argument("-fn2", type=str, default="features_2.csv",
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


    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)

    X, y = None, None
    do_eval = False
    ytrue = []

    if os.path.exists(args.fn1):
        df = pd.read_csv(args.fn1, index_col=0, header=0).dropna()
        if args.train:
            y = df[args.pKa_col[0]].values
        X = df.values[:, :3840]

        if args.pKa_col[0] in df.columns.values:
            ytrue = df[args.pKa_col[0]].values

            do_eval = True

    if os.path.exists(args.fn2):
        df2 = pd.read_csv(args.fn2, index_col=0, header=0).dropna()
        X2 = df2.values[:, :3840]
        if args.train:
            y2 = df2[args.pKa_col[-1]].values
            y = list(y) + list(y2)

        X = np.concatenate((X, X2), axis=0)

        if args.pKa_col[-1] in df2.columns.values:
            ytrue2 = df2[args.pKa_col[1]].values

            ytrue = list(ytrue) + list(ytrue2)

            do_eval = True

    print("DataSet Loaded")

    if args.train > 0:

        scaler = preprocessing.StandardScaler()
        Xs = scaler.fit_transform(X)
        joblib.dump(scaler, args.scaler)
        print("DataSet Scaled")

        Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xs, y, test_size=0.2)
        print("Train and test split")
        Xtrain = Xtrain.reshape((-1, 64, 60, 1))
        model = create_model((64, 60, 1), lr=args.lr_init)

        # callbacks
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='auto',)
        logger = tf.keras.callbacks.CSVLogger(args.log, separator=',', append=False)
        bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath="bestmodel_"+args.model, verbose=1, save_best_only=True)

        # train the model
        history = model.fit(Xtrain, ytrain, validation_data=(Xtest.reshape(-1, 64, 60, 1), ytest),
                            batch_size=64, epochs=200, verbose=1, callbacks=[stop, logger, bestmodel])

        model.save(args.model)
        print("Save model. ")

        #np_hist = np.array(history)
        #np.savetxt(args.history, np_hist, delimiter=",", fmt="%.4f")
        #print("Save history.")

    else:
        scaler = joblib.load(args.scaler)

        Xs = scaler.transform(X).reshape((-1, 64, 60, 1))

        model = tf.keras.models.load_model(args.model,
                                           custom_objects={'RMSE': RMSE,
                                                           'PCC': PCC,
                                                           'PCC_RMSE':PCC_RMSE})

        ypred = pd.DataFrame()
        ypred['pKa_predicted'] = model.predict(Xs).ravel()
        if do_eval:
            print("PCC : %.3f" % pcc(ypred['pKa_predicted'].values, ytrue))
            print("RMSE: %.3f" % rmse(ypred['pKa_predicted'].values, ytrue))

            ypred['pKa_true'] = ytrue

        ypred.to_csv(args.out, header=True, index=True, float_format="%.3f")



