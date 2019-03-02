#!/usr/bin/env python

import numpy as np
import mdtraj as mt
import sys
from deepunion import region_mutate


def rmsd(mol1, mol2):

    #cpdb = region_mutate.coordinatesPDB()
    #with open(mol1) as lines:
    #    m1 = cpdb.getAtomCrdFromLines([x for x in lines if ("ATOM" in x or "HETATM" in x)])
    #with open(mol2) as lines:
    #    m2 = cpdb.getAtomCrdFromLines([x for x in lines if ("ATOM" in x or "HETATM" in x)])
    m1 = mt.load(mol1).xyz[0]
    m2 = mt.load(mol2).xyz[0]

    if m1.shape != m2.shape:
        print("ERROR: Atom numbers are not the same in the two files. Check your input.")
        print("ERROR: Exit now!")
        sys.exit(0)

    rmsd = np.sum((m1 - m2).ravel() ** 2 / m1.shape[0])

    return np.sqrt(rmsd)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Make sure that the atoms' sequences in both files are the same exactly. ")
        print("Usage: python rmsd.py native_pose.pdb docked_pose.pdb")
        print("Only pdb files are accepted as inputs. ")
        sys.exit(0)
    else:
        print(rmsd(sys.argv[1], sys.argv[2]))

