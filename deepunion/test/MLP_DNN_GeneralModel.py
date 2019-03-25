import sys
from sklearn import preprocessing, model_selection
import tensorflow as tf
import pandas as pd


def create_model(input_size):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(100, input_shape=input_size, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(200, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(400, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1600, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, activation="relu"))

    model.compile(optimizer="sgd", loss="mse")

    return model

if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1], header=True)

    y = df['logS']
    X = df.values[:, :-3]

    scaler = preprocessing.StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xs, y, test_size=0.2)

    model = create_model((Xtrain.shape[1], ))

    model.fit(Xtrain, ytrain, batch=128, epochs=400, verbose=1)

    # save model
    model.save("DNN_MLP_solubility.h5")
