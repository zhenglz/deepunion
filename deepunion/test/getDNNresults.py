#!/usr/bin/env python

import pandas as pd
import argparse
import sys, os

if __name__ == "__main__":

    d = """"""
    parser = argparse.ArgumentParser(description=d)
    parser.add_argument("-fn", type=str, nargs="+",
                        help="Input. The log files from TF.")
    parser.add_argument("-patience", default=40, type=int,
                        help="Input. The patience for early stopping.")
    parser.add_argument("-epochs", type=int, default=200,
                        help="Input. The number of epochs in training. ")
    parser.add_argument("-filters", type=str, default="128+64+32")
    args = parser.parse_args()

    EPOCHS = args.epochs
    p = args.patience + 1

    fn_list = args.fn
    for i, fn in enumerate(fn_list):
        if i == 0 :
            print("Pooling  Batch  Dropout  Alpha  filters  Loss(T)  RMSE(T)  PCC(T)  Loss(V)   RMSE(V)   PCC(V)")

        if not os.path.exists(fn):
            print("TF log file %s not exists. "%fn)
            #sys.exit(0)
        else:
            df = pd.read_csv(fn, header=0, index_col=0)

            batch = fn.split("batch")[1].split("_")[0]
            dropout = fn.split("dropout")[1].split("_")[0]
            alpha = fn.split("alpha")[1][:3]
            if "with" in fn:
                pooling = "yes"
            else:
                pooling = "no"

            if df.index.values[-1] == EPOCHS or df.shape[0] < args.patience + 1:
                print(fn, "Model training per-terminated before a final solution fixed. ")
                #sys.exit(0)
            else:
                to_print = "%24s, %6s, %6s, %6s, %6s, %12s," % (fn, pooling, batch, dropout, alpha, args.filters)
                #dat = df.iloc[-1*patience, :]
                to_print += "%8.4f,%8.4f,%8.4f,%8.4f,%8.4f, %8.4f" %(df['loss'].values[-p], df['rmse_train'].values[-p], df['pcc_train'].values[-p],
                                                                     df['loss_val'].values[-p], df['rmse_val'].values[-p], df['pcc_val'].values[-p])

                print(to_print)

