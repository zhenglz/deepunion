import numpy as np
import pandas as pd
import mdtraj as mt
import itertools
import sys
from collections import OrderedDict

class AtomTypeCounts(object):

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
            self.distance_matrix_ = mt.compute_distances(self.pdb, atom_pairs=all_pairs)[]

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

    for cutoff in ncutoffs:
        cplx.cutoff_count(cutoff)
        counts.append(cplx.counts_)

    results = []

    for n in range(len(ncutoffs)):
        #count_dict = dict.fromkeys(keys, 0.0)
        d = OrderedDict()
        d = d.fromkeys(keys, 0.0)
        for e_e, c in rec_lig_element_combines, counts[n]:
            d[e_e] += c

        results += d.values()

    return results, keys


if __name__ == "__main__":

    with open(sys.argv[1]) as lines:
        inputs = [x.split()[:2] for x in lines if "#" not in s]

    out = sys.argv[2]

    n_cutoffs = np.linspace(0.2, 1.2, 10)

    with open(fn) as lines:
        lines = [x for x in lines if ("#" not in x and len(x.split()) >= 2)].copy()
        inputs = [x.split()[:2] for x in lines]

    results = []

    ele_pairs =[]

    success = []

    for p in inputs:
        fn = p[0]
        lig_code = p[1]

        try:
            r, ele_pairs = generate_features(fn, lig_code, n_cutoffs)
            results.append(r)
            success.append(1.)

        except:
            r = results[-1]
            success.append(0.)

    df = pd.DataFrame(results)
    df.index = [x[0] for x in inputs]
    df.columns = ele_pairs * len(n_cutoffs)

    df.to_csv(out, sep=",", float_format="%.1f")

    print("Complete calculations. ")

