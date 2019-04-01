#!/usr/bin/env python

import sys
import pandas as pd
import os
import argparse

if __name__ == "__main__":
    d = """
    Cat csv files and merge the header line and use the first column as index.
    """

    parser = argparse.ArgumentParser(description=d)
    parser.add_argument("-fn", default=["training_*.csv"], type=str, nargs="+",
                        help="Input, optional. The input file list.")
    parser.add_argument("-out", default="output.csv", type=str,
                        help="Output, optional. The output file name.")
    parser.add_argument("-target", default="training.csv", type=str,
                        help="Input, optional. The target input file name.")

    args = parser.parse_args()
    if len(sys.argv) <= 3:
        parser.print_help()
        sys.exit(0)

    # PWD
    os.chdir(os.getcwd())

    # target filename
    if os.path.exists(args.target):
        target = pd.read_csv(args.target, index_col=0, header=0)
        print(target.shape)
    else:
        target = pd.DataFrame()

    dat = pd.DataFrame()
    for i, fn in enumerate(args.fn):
        df = pd.read_csv(fn, index_col=0, header=0)
        if i == 0:
            dat = df
        else:
            dat = pd.concat([dat, df])

        print(dat.shape)
    #dat.index = [x.split("/")[2] for x in dat.index]

    if os.path.exists(args.target):
        if dat.shape[0] != target.shape[0]:
            print("Number of samples unequal, exit now!")
            sys.exit(1)
        else:
            dat['pKa'] = target['pKa'].values
            #dat = dat.reindex(target.index.values)

    dat.to_csv(args.out, header=True, index=True)


