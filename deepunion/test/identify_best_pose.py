#!/usr/bin/env python

#from deepunion import rmsd
from deepunion import babel_converter
import subprocess as sp
from glob import glob
import sys
import numpy as np
#import pandas as pd
from deepunion import region_mutate
#import os
#from deepunion import region_mutate
import mdtraj as mt

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
        print("ERROR: ", mol1, mol2)
        print("ERROR: Exit now!")
        sys.exit(0)

    rmsd = np.sum((m1 - m2).ravel() ** 2 / m1.shape[0])

    return np.sqrt(rmsd)

def converter_multiple(inp, out, multiple=True):

    if multiple:
        cmd = "obabel %s -O %s -m" % (inp, out)
        job = sp.Popen(cmd, shell=True)
        job.communicate()
    else:
        babel_converter(inp, out, mode="general")

def calculate_rmsd(ref="reference.pdb"):

    rmsd_list = []
    fn_list = glob("*_vinaout_*.pdb")

    for fn in fn_list:
        r = rmsd(ref, fn)
        rmsd_list.append((fn, r))

    return rmsd_list

if __name__ == "__main__":
    prefix = sys.argv[1]
    rmsd_cutoff = 0.3

    #rmsd_df = pd.DataFrame()

    # split vina output
    converter_multiple(prefix+"_vinaout.pdbqt", prefix+"_vinaout_.pdb")

    # get a reference, the crystal position of a ligand
    babel_converter(prefix+"_ligand.mol2.pdbqt", "temp_ligand.pdb", mode="general")
    job = sp.Popen("awk '$1 ~ /ATOM/ || $1 ~ /HETATM/ {print $0}' temp_ligand.pdb > reference_ligand.pdb", shell=True)
    job.communicate()

    # get the rmsd list of a ligand
    rmsds = calculate_rmsd("reference_ligand.pdb")

    # get the best rmsd
    rmsds_sorted = sorted(rmsds, key=lambda x: x[1], reverse=False)
    print(rmsds_sorted)

    if rmsds_sorted[0][1] <= rmsd_cutoff:
        lig2combine = rmsds_sorted[0][0]

        # change ligand resname to LIG
        rpdb = region_mutate.rewritePDB(lig2combine)
        tofile = open(lig2combine+"_rew", "w")
        with open(lig2combine) as lines:
            for s in [x for x in lines if "ATOM" in x]:
                new_line = rpdb.resNameChanger(s, "LIG")
                tofile.write(new_line)
        tofile.close()

        cmd = "cat %s %s | awk '$1 ~ /ATOM/ {print $0}' > %s " \
              % (prefix+"_protein.pdb",
                 lig2combine+"_rew",
                 prefix+"_bestpose_complex.pdb")

        job = sp.Popen(cmd, shell=True)
        job.communicate()

        sp.Popen("echo %s %.3f %s > best_rmsd "% (prefix, rmsds_sorted[0][1], rmsds_sorted[0][0]), shell=True)
