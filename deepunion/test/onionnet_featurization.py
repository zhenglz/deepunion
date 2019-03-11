#!/usr/bin/env python

import numpy as np
import pandas as pd
import mdtraj as mt
import itertools
import sys
from collections import OrderedDict
from mpi4py import MPI


class AtomTypeCounts(object):
    """Featurization of Protein-Ligand Complex based on
    onion-shape distance counts of atom-types.

    Parameters
    ----------
    pdb_fn : str
        The input pdb file name.
    lig_code : str
        The ligand residue name in the input pdb file.

    Attributes
    ----------
    pdb : mdtraj.Trajectory
        The mdtraj.trajectory object containing the pdb.
    receptor_indices : np.ndarray
        The receptor (protein) atom indices in mdtraj.Trajectory
    ligand_indices : np.ndarray
        The ligand (protein) atom indices in mdtraj.Trajectory
    rec_ele : np.ndarray
        The element types of each of the atoms in the receptor
    lig_ele : np.ndarray
        The element types of each of the atoms in the ligand
    lig_code : str
        The ligand residue name in the input pdb file
    pdb_parsed_ : bool
        Whether the pdb file has been parsed.
    distance_computed : bool
        Whether the distances between atoms in receptor and ligand has been computed.
    distance_matrix_ : np.ndarray, shape = [ N1 * N2, ]
        The distances between all atom pairs
        N1 and N2 are the atom numbers in receptor and ligand respectively.
    counts_: np.ndarray, shape = [ N1 * N2, ]
        The contact numbers between all atom pairs
        N1 and N2 are the atom numbers in receptor and ligand respectively.

    """

    def __init__(self, pdb_fn, lig_code):

        self.pdb = mt.load(pdb_fn)

        self.receptor_indices = np.array([])
        self.ligand_indices = np.array([])

        self.rec_ele = np.array([])
        self.lig_ele = np.array([])

        self.lig_code = lig_code

        self.pdb_parsed_ = False
        self.distance_computed_ = False

        self.distance_matrix_ = np.array([])
        self.counts_ = np.array([])

    def parsePDB(self, rec_sele="protein", lig_sele="resname LIG"):

        top = self.pdb.topology

        self.receptor_indices = top.select(rec_sele)
        self.ligand_indices = top.select(lig_sele)

        table, bond = top.to_dataframe()

        self.rec_ele = table['element'][self.receptor_indices]
        self.lig_ele = table['element'][self.ligand_indices]

        self.pdb_parsed_ = True

        return self

    def distance_pairs(self):

        if not self.pdb_parsed_:
            self.parsePDB()

        all_pairs = itertools.product(self.receptor_indices, self.ligand_indices)

        if not self.distance_computed_:
            self.distance_matrix_ = mt.compute_distances(self.pdb, atom_pairs=all_pairs)[0]

        self.distance_computed_ = True

        return self

    def cutoff_count(self, cutoff=0.35):

        self.counts_ = (self.distance_matrix_ <= cutoff) * 1.0

        return self


def generate_features(complex_fn, lig_code, ncutoffs):

    all_elements = ["H", "C", "O", "N", "P", "S", "Br", "Du"]
    keys = ["_".join(x) for x in list(itertools.product(all_elements, all_elements))]

    cplx = AtomTypeCounts(complex_fn, lig_code)
    cplx.parsePDB(rec_sele="protein", lig_sele="resname %s" % lig_code)

    lig = cplx.lig_ele
    rec = cplx.rec_ele

    new_lig, new_rec = [], []
    for e in lig:
        if e not in all_elements:
            new_lig.append("Du")
        else:
            new_lig.append(e)
    for e in rec:
        if e not in all_elements:
            new_rec.append("Du")
        else:
            new_rec.append(e)

    rec_lig_element_combines = ["_".join(x) for x in list(itertools.product(new_rec, new_lig))]
    cplx.distance_pairs()

    counts = []

    onion_counts = []

    for i, cutoff in enumerate(ncutoffs):
        cplx.cutoff_count(cutoff)

        if i == 0:
            onion_counts.append(cplx.counts_)
        else:
            onion_counts.append(cplx.counts_ - counts[-1])

        counts.append(cplx.counts_)

    results = []

    for n in range(len(ncutoffs)):
        #count_dict = dict.fromkeys(keys, 0.0)
        d = OrderedDict()
        d = d.fromkeys(keys, 0.0)
        for e_e, c in zip(rec_lig_element_combines, onion_counts[n]):
            d[e_e] += c

        results += d.values()

    return results, keys


if __name__ == "__main__":
    #print("Start Now ... ")
    inp = sys.argv[1]
    lig = "LIG"
    out = sys.argv[2]
    n_cutoffs = np.linspace(0.1, 3.05, 60)

    results = []
    ele_pairs =[]

    try:
        results, ele_pairs = generate_features(inp, lig, n_cutoffs)

    except:
        #r = results[-1]
        results = list([0., ]*3840) + [0.0, ]

    #results = [inp, ] + results
    print(len(results))
    with open(out, "w") as tofile:
        l = inp+","+",".join(format(x, ".3f") for x in results)+"\n"
        tofile.write(l)
