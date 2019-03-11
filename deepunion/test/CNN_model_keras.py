import sys
from sklearn import preprocessing, model_selection
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import argparse
from argparse import RawTextHelpFormatter


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


def pcc(y_true, y_pred):

    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


def create_model(input_size):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(128, 4, 1, input_shape=input_size))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(64, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(32, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(2000, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1000,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(400, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(40, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
    model.add(tf.keras.layers.Activation("relu"))

    sgd = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6, )
    model.compile(optimizer=sgd, loss=PCC_RMSE, metrics=["mse", pcc, RMSE])

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-fn1", type=str,
                        help="Input. The docked cplx feature set.")
    parser.add_argument("-fn2", type=str,
                        help="Input. The PDBBind feature set.")
    parser.add_argument("-history", type=str, default="history.csv",
                        help="Output. The history information. ")
    parser.add_argument("-pKa_col", type=str, default="pKa_relu",
                        help="Input. The pKa colname as the target. ")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
    parser.add_argument("-model", type=str, default="DNN_Model.h5",
                        help="Output. The trained DNN model file to save. ")
    parser.add_argument("-log", type=str, default="logger.csv",
                        help="Output. The logger file name to save. ")

    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)

    df = pd.read_csv(args.fn1, index_col=0, header=0).dropna()

    y = df[args.pKa_col].values
    X = df.values[:, :3840]

    df2 = pd.read_csv(args.fn2, index_col=0, header=0).dropna()
    X2 = df2.values[:, :3840]

    y = list(y) + list(df2.values[:, -1])
    X = np.concatenate((X, X2), axis=0)

    print("DataSet Loaded")
 
    scaler = preprocessing.StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, args.scaler)
    print("DataSet Scaled")

    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xs, y, test_size=0.2)
    print("Train and test split")
    Xtrain = Xtrain.reshape((-1, 64, 60, 1))
    model = create_model((64, 60, 1))

    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='auto',)
    logger = tf.keras.callbacks.CSVLogger(args.log, separator=',', append=False)
    bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath="bestmodel_"+args.model, verbose=1, save_best_only=True)

    history = model.fit(Xtrain, ytrain, validation_data=(Xtest.reshape(-1, 64, 60, 1), ytest),
                        batch_size=64, epochs=200, verbose=1, callbacks=[stop, logger, bestmodel])

    model.save(args.model)
    print("Save model. ")

    np_hist = np.array(history)
    np.savetxt(args.history, np_hist, delimiter=",", fmt="%.4f")
    print("Save history.")

