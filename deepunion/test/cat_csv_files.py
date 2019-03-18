#!/usr/bin/env python

import sys
from glob import glob
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="cat csv files")
    parser.add_argument("-fn", default="training_*.csv", type=str,
                        help="Input, optional. The input file list.")
    parser.add_argument("-out", default="output.csv", type=str,
                        help="Output, optional. The output file name.")
    parser.add_argument("-target", default="training.csv", type=str,
                        help="Input, optional. The target input file name.")

    args = parser.parse_args()

    # target filename
    target = pd.read_csv(args.target, index_col=0, header=0)
    print(target.shape)

    fns = glob(args.fn)

    dat = pd.DataFrame()
    for i, fn in enumerate(fns):
        df = pd.read_csv(fn, index_col=0, header=0)
        if i == 0:
            dat = df
        else:
            dat = pd.concat([dat, df])

        print(dat.shape)

    dat.index = [x.split("/")[2] for x in dat.index]

    if dat.shape[0] != target.shape[0]:
        print("Number of samples unequal, exit now!")
        sys.exit(1)

    dat = dat.reindex(target.index.values)
    dat['pKa'] = target['pKa'].values

    dat.to_csv(args.out, header=True, index=True)


