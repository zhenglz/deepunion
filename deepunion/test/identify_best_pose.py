#!/usr/bin/env python

from deepunion import rmsd
from deepunion import babel_converter
import subprocess as sp
from glob import glob
import sys
import pandas as pd
from deepunion import region_mutate
import os

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
        #print(ref, fn)
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
    with open("rmsd_ranking_list", "w") as tofile:
        for i in range(len(rmsds_sorted)):
            tofile.write("%20s %6.3f \n" % (rmsds_sorted[i][0], rmsds_sorted[i][1]))

    if rmsds_sorted[0][1] <= rmsd_cutoff and not os.path.exists(prefix+"_vinaout_bestpose_complex.pdb"):
        lig2combine = rmsds_sorted[0][0]

        # change ligand resname to LIG
        rpdb = region_mutate.rewritePDB(lig2combine)
        tofile = open(lig2combine+"_rew", "w")
        with open(lig2combine) as lines:
            for s in [x for x in lines if "ATOM" in x]:
                new_line = rpdb.resNameChanger(s, "LIG")
                tofile.write(new_line)
        tofile.close()

        cmd = "cat %s %s > %s" % (prefix+"_protein.pdb",
                                  lig2combine+"_rew",
                                  prefix+"_vinaout_bestpose_complex.pdb")

        job = sp.Popen(cmd, shell=True)
        job.communicate()
          
        sp.Popen("echo %s %.3f > best_rmsd "% (prefix, rmsds_sorted[0][1]), shell=True)
