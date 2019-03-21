#!/usr/bin/env python

import os, sys
from rdkit import Chem
import subprocess as sp
import dockml.pdbIO as pdbio
from deepunion import builder


def convert_lig(lig_in, lig_out):

    try:
        mol = Chem.MolFromMol2File(lig_in, removeHs=False)
        Chem.MolToPDBFile(mol, lig_out)
    except:
        print("Switch to open babel. ")
        builder.babel_converter(lig_in, lig_out)

    return None

def cat_rec_lig(rec, lig, out):
    job = sp.Popen("cat %s | awk '$1 ~ /ATOM/ {print $0}' > temp_rec" %rec, shell=True )
    job.communicate()
    job = sp.Popen("cat temp_rec %s | awk '$1 ~ /ATOM/ || $1 ~ /HETATM/ {print $0}' | awk '$4 != /HOH/ {print $0}' > %s"%(lig, out), shell=True)
    job.communicate()

def lig_name_change(lig_in, lig_out, lig_code):

    pio = pdbio.rewritePDB(lig_in)
    tofile = open(lig_out, "w")
    with open(lig_in) as lines:
        for s in lines:
            if len(s.split()) and s.split()[0] in ['ATOM', 'HETATM']:
                nl = pio.resNameChanger(s, lig_code)
                n2 = pio.chainIDChanger(nl, "Z")
                tofile.write(n2)

    tofile.close()
    return None

def main():

    inputs = [x.split()[0] for x in open(sys.argv[1]).readlines() if "#" not in x]
    print(inputs)
    for p in inputs:

        rec = os.path.join(p, p+"_protein.pdb")
        lig = os.path.join(p, p+"_ligand.mol2")
        if True: #if not os.path.exists(os.path.join(p, "%s_cplx.pdb" % p)):
            try:
                convert_lig(lig, "t1_%s.pdb" % p)
                lig_name_change("t1_%s.pdb" % p, "t2_%s.pdb" % p, "LIG")
                cat_rec_lig(rec, "t2_%s.pdb" % p, os.path.join(p, "%s_cplx.pdb" % p))
            except:
                print("Not successful : ", p)

#        print(p)

main()

