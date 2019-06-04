from deepunion import builder
import itertools
import numpy as np
from psikit import Psikit


def mol_reader(smile):

    mbuild = builder.CompoundBuilder()
    mbuild.load_mol(smile)
    mbuild.generate_conformer()

    return mbuild.molecule_


def get_coordinates(m):

    m.GetConformer()
    return m.GetPositions()


def distance(pair):

    return np.sqrt(np.sum(np.square(pair[0] - pair[1])))


def cmap(coords):

    pairs = itertools.product(coords)
    distances = list(map(distance, pairs))

    return distances

def resp_charges(smile, bset="hf/6-31g"):
    # https://github.com/Mishima-syk/psikit

    pk = Psikit.read_from_smiles(smiles_str=smile)

    pk.energy()
    pk.optimize(basis_sets=bset)

    return pk.cal_resp_charges()

