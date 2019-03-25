#!/usr/bin/env python

import numpy as np
import pandas as pd
import mdtraj as mt
import itertools
import re
import sys
from collections import OrderedDict
from mpi4py import MPI
import argparse
from argparse import RawDescriptionHelpFormatter


class ResidueCounts(object):

    def __init__(self, pdb_fn, ligcode="LIG"):

        self.pdb = mt.load_pdb(pdb_fn)
        self.receptor_ids_ = None
        self.ligand_ids_ = None
        self.resid_pairs_ = None
        self.ligand_n_atoms_ = 0
        self.distance_calculated_ = False
        self.prepared_ = False
        self.distances_all_pairs_ = None

        self.max_pairs_ = 500

        self.top = self.pdb.topology

    def get_receptor_seq(self):

        pattern = re.compile("[A-Za-z]*")

        res_seq = [str(x) for x in list(self.top.residues)[:-1]]

        self.seq = [pattern.match(x).group(0) for x in res_seq]

        self.receptor_ids_ = np.arange(len(self.seq))
        self.ligand_ids_ = np.array([self.receptor_ids_[-1]+1, ])

        #print(self.receptor_ids_, self.ligand_ids_)

        self.ligand_n_atoms_ = self.top.select("resid %d" % len(self.seq)).shape[0]

        return self

    def get_resid_pairs(self):

        pairs_ = list(itertools.product(self.receptor_ids_, self.ligand_ids_))
        if len(pairs_) > self.max_pairs_:
            self.resid_pairs_ = pairs_[:self.max_pairs_]
        else:
            self.resid_pairs_ = pairs_

        self.resid_pairs_ = np.array(self.resid_pairs_)

        return self

    def contact_calpha(self, cutoff):
        # define pairs
        c_alpha_indices = self.top.select("name CA and (not resname LIG)")
        print(c_alpha_indices.shape)
        pairs_ = list(itertools.product(c_alpha_indices, c_alpha_indices))

        distance_matrix_ = mt.compute_distances(self.pdb, atom_pairs=pairs_)[0]
        distance_matrix_ = distance_matrix_.reshape((-1, int(np.sqrt(distance_matrix_.shape[0]))))

        cmap = (distance_matrix_ <= cutoff)*1.0

        return np.sum(cmap, axis=0)

    def cal_distances(self, residue_pair, ignore_hydrogen=True):

        if ignore_hydrogen:
            indices_a = self.pdb.topology.select("(resid %d) and (symbol != H)" % residue_pair[0])
            indices_b = self.pdb.topology.select("(resid %d) and (symbol != H)" % residue_pair[1])
        else:
            indices_a = self.pdb.topology.select("resid %d" % residue_pair[0])
            indices_b = self.pdb.topology.select("resid %d" % residue_pair[1])

        pairs = itertools.product(indices_a, indices_b)

        return mt.compute_distances(self.pdb, pairs)[0]

    def contacts_nbyn(self, cutoff, resid_pair):

        # if not self.distance_calculated_:
        distances = np.sum(self.cal_distances(resid_pair) <= cutoff)
        nbyn = np.sqrt(self.top.select("resid %d" % resid_pair[0]).shape[0] * self.ligand_n_atoms_)

        return distances / nbyn

    def do_preparation(self):
        if self.receptor_ids_ is None:
            self.get_receptor_seq()
        if self.resid_pairs_ is None:
            self.get_resid_pairs()

        self.prepared_ = True

        return self

    def distances_all_pairs(self, cutoff, verbose=True):
        # do preparation
        if not self.prepared_:
            self.do_preparation()

        # looping over all pairs
        d = np.zeros(len(self.resid_pairs_))
        for i, p in enumerate(self.resid_pairs_):
            if i % 100 == 0 and verbose:
                print("Progress of residue-ligand contacts: ", i)

            d[i] = self.contacts_nbyn(cutoff, p)

        self.distances_all_pairs_ = d
        return self


def distance_padding(dist, max_pairs_=500, padding_with=0.0):
    """

    Parameters
    ----------
    dist: np.array, shape = [N, ]
        The input data array
    max_pairs_: int, default = 500
        The maximium number of features in the array
    padding_with: float, default=0.0
        The value to pad to the array

    Returns
    -------
    d: np.array, shape = [N, ]
        The returned array after padding
    """

    if dist.shape[0] < max_pairs_:
        d = np.concatenate((dist, np.repeat(padding_with, max_pairs_ - dist.shape[0])))
    elif dist.shape == max_pairs_:
        d = dist
    else:
        d = dist[:max_pairs_]
        print("Warning: number of features higher than %d" % max_pairs_)

    return d


def hydrophobicity():
    '''http://assets.geneious.com/manual/8.0/GeneiousManualsu41.html'''
    hydrophobic = {
        'PHE': 1.0,
        'LEU': 0.943,
        'ILE': 0.943,
        'TYR': 0.880,
        'TRP': 0.878,
        'VAL': 0.825,
        'MET': 0.738,
        'PRO': 0.711,
        'CYS': 0.680,
        'ALA': 0.616,
        'GLY': 0.501,
        'THR': 0.450,
        'SER': 0.359,
        'LYS': 0.283,
        'GLN': 0.251,
        'ASN': 0.236,
        'HIS': 0.165,
        'GLU': 0.043,
        'ASP': 0.028,
        'ARG': 0.0,
        'UNK': 0.501,
    }

    return hydrophobic


def polarizability():
    """https://www.researchgate.net/publication/220043303_Polarizabilities_of_amino_acid_residues/figures"""
    polar = {
        'PHE': 121.43,
        'LEU': 91.6,
        'ILE': 91.21,
        'TYR': 126.19,
        'TRP': 153.06,
        'VAL': 76.09,
        'MET': 102.31,
        'PRO': 73.47,
        'CYS': 74.99,
        'ALA': 50.16,
        'GLY': 36.66,
        'THR': 66.46,
        'SER': 53.82,
        'LYS': 101.73,
        'GLN': 88.79,
        'ASN': 73.15,
        'HIS': 99.35,
        'GLU': 84.67,
        'ASP': 69.09,
        'ARG': 114.81,
        'UNK': 36.66,
    }

    return polar


def stringcoding():
    """Sequence from http://www.bligbi.com/amino-acid-table_242763/epic-amino-acid-table-l99-
    on-nice-home-designing-ideas-with-amino-acid-table/"""

    sequence = {
        'PHE': 18,
        'LEU': 16,
        'ILE': 15,
        'TYR': 19,
        'TRP': 20,
        'VAL': 14,
        'MET': 17,
        'PRO': 12,
        'CYS': 10,
        'ALA': 13,
        'GLY': 11,
        'THR': 7,
        'SER': 6,
        'LYS': 3,
        'GLN': 8,
        'ASN': 8,
        'HIS': 2,
        'GLU': 4,
        'ASP': 5,
        'ARG': 1,
        'UNK': 11,
    }

    return sequence


def residue_string2code(seq, method=stringcoding):
    mapper = method()
    return [mapper[x] if x in mapper.keys()
            else mapper['UNK']
            for x in seq]


def generate_contact_features(complex_fn, ncutoffs, verbose=True):

    rescont = ResidueCounts(complex_fn)

    if verbose: print("START preparation")
    rescont.do_preparation()
    if verbose: print("COMPLETE preparation")

    seq = rescont.seq
    if verbose: print("Length of residues ", len(seq))

    if verbose: print("START alpha-C contact map")
    r = np.array([])
    for c in np.linspace(0.3, 1.2, 4):
        cmap = rescont.contact_calpha(cutoff=c)
        cmap = distance_padding(cmap)
        r = np.concatenate((r, cmap))
        if verbose: print(cmap)
    if verbose:print("COMPLETE contactmap")

    for m in [stringcoding, polarizability, hydrophobicity]:
        coding = np.array(residue_string2code(seq, m))
        if verbose: print("START sequence to coding")
        mapper = m()
        coding = distance_padding(coding, padding_with=mapper['GLY'])
        if verbose: print(coding)
        r = np.concatenate((r, coding))

    if verbose:print("COMPLETE sequence to coding")
    if verbose:print("SHAPE of result: ", r.shape)

    for c in ncutoffs:
        if verbose: print("START residue based atom contact nbyn, cutoff=", c)
        rescont.distances_all_pairs(c, verbose)
        d = distance_padding(rescont.distances_all_pairs_)
        if verbose: print(d)
        r = np.concatenate((r, d))

    if verbose: print("SHAPE of result: ", r.shape)

    return r


if __name__ == "__main__":

    print("Start Now ... ")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    d = """
    Predicting protein-ligand binding affinities (pKa) with OnionNet model. 
    Citation: Coming soon ... ...
    Author: Liangzhen Zheng

    This script is used to generate inter-molecular element-type specific 
    contact features. Installation instructions should be refered to 
    https://github.com/zhenglz/onionnet

    Examples:
    Show help information
    python generate_features.py -h

    Run the script with one CPU core
    python generate_features.py -inp input_samples.dat -out features_samples.csv

    Run the script with MPI 
    mpirun -np 12 python generate_contact_features.py -inp input_testing.dat -out testing_features_rescount.csv 
                         -start 0.4 -end 2.0 -n_shells 5 -v 0

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp", type=str, default="input.dat",
                        help="Input. The input file containg the file path of each \n"
                             "of the protein-ligand complexes files (in pdb format.)\n"
                             "There should be only 1 column, each row or line containing\n"
                             "the input file path, relative or absolute path.")
    parser.add_argument("-out", type=str, default="output.csv",
                        help="Output. Default is output.csv \n"
                             "The output file name containing the features, each sample\n"
                             "per row. ")
    parser.add_argument("-lig", type=str, default="LIG",
                        help="Input, optional. Default is LIG. \n"
                             "The ligand molecule residue name (code, 3 characters) in the \n"
                             "complex pdb file. ")
    parser.add_argument("-start", type=float, default=0.1,
                        help="Input, optional. Default is 0.05 nm. "
                             "The initial shell thickness. ")
    parser.add_argument("-end", type=float, default=3.0,
                        help="Input, optional. Default is 3.05 nm. "
                             "The boundary of last shell.")
    parser.add_argument("-delta", type=float, default=0.05,
                        help="Input, optional. Default is 0.05 nm. "
                             "The thickness of the shells.")
    parser.add_argument("-n_shells", type=int, default=60,
                        help="Input, optional. Default is 60. "
                             "The number of shells for featurization. ")
    parser.add_argument("-v", default=1, type=int,
                        help="Input, optional. Default is 1. "
                             "Whether output detail information.")

    args = parser.parse_args()

    if rank == 0:
        if len(sys.argv) < 3:
            parser.print_help()
            sys.exit(0)

        # spreading the calculating list to different MPI ranks
        with open(args.inp) as lines:
            lines = [x for x in lines if ("#" not in x and len(x.split()) >= 1)].copy()
            inputs = [x.split()[0] for x in lines]

        inputs_list = []
        aver_size = int(len(inputs) / size)
        if args.v:
            print(size, aver_size)
        for i in range(size - 1):
            inputs_list.append(inputs[int(i * aver_size):int((i + 1) * aver_size)])
        inputs_list.append(inputs[(size - 1) * aver_size:])

    else:
        inputs_list = None

    inputs = comm.scatter(inputs_list, root=0)

    # defining the shell structures ... (do not change)
    n_cutoffs = list(np.linspace(args.start, args.end, args.n_shells))
    if args.v:
        print(n_cutoffs)

    results = []
    ele_pairs = []

    # computing the features now ...
    for p in inputs:

        try:
            # the main function for featurization ...
            r= generate_contact_features(p, n_cutoffs, verbose=args.v)
            print(rank, p)

        except RuntimeError:
            r = [0., ] * 500 * (args.n_shells + 4 + 3)
            print(rank, "Not successful. ", p)

        results.append(r)

    # saving features to a file now ...
    df = pd.DataFrame(results)
    try:
        df.index = inputs
    except:
        df.index = np.arange(df.shape[0])

    col_n = ["F"+ str(x) for x in range(df.shape[1])]
    df.columns = col_n
    df.to_csv("rank%d_" % rank + args.out, sep=",", float_format="%.4f", index=True)

    print(rank, "Complete calculations. ")

