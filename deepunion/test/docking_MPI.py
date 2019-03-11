#!/usr/bin/env python

import sys, os
from mpi4py import MPI
from deepunion import docking, builder

"""
Docking Routine

perform docking with vina with MPI 

"""


def do_docking():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        keep_codes = []
        with open(sys.argv[1]) as lines:
            pdb_codes = [x.strip("\n") for x in lines]
        #print(pdb_codes)

        for c in pdb_codes:
            if not os.path.exists("%s/%s_vinaout.pdbqt"%(c, c)) and os.path.exists("%s/%s_protein.pdb.pdbqt"%(c, c)):
                keep_codes.append(c)
        print(keep_codes)
        chunk = int(len(keep_codes) / size)

        input_lists = []
        for i in range(size-1):
            input_lists.append(keep_codes[i*chunk: i*chunk+chunk])
        input_lists.append(keep_codes[(size-1)*chunk:])
    else:
        input_lists = None

    inputs = comm.scatter(input_lists, root=0)

    for c in inputs:

        rec = "%s/%s_protein.pdb.pdbqt" % (c, c)

        #process the ligand
        lig = "%s/%s_ligand.mol2" % (c, c)
        docking.pdb2pdbqt(lig, lig + ".pdbqt", )
        docking.pdb2pdbqt(lig, lig + ".pdb", keep_polarH=False)

        if os.path.exists(rec) and os.path.exists(lig+".pdbqt"):

            try:
                out = "%s/%s_vinaout.pdbqt" %(c, c)
                log = "%s/log_vina.log" % c

                config = "%s/vina.config" % c

                if not os.path.exists(out):
                    rec_prep = docking.ReceptorPrepare(rec)
                    # rec_prep.receptor_addH("H_"+rec)
                    xyz_c = rec_prep.pocket_center(LIG=lig + ".pdb")

                    # docking.pdb2pdbqt(rec, "temp.pdbqt")
                    # job = sp.Popen("awk '$1 ~ /ATOM/ {print $0}' temp.pdbqt > %s.pdbqt" % rec, shell=True)
                    # job.communicate()
                    vina = docking.VinaDocking()
                    vina.vina_config(rec, lig + ".pdbqt", out, 32, 8, xyz_c, [30, 30, 30], log,
                                     n_modes=20, config=config)
                    vina.run_docking()

                print("COMPLETE on rank %d: %s" %(rank, c))
            except:
                print("FAIL     on rank %d: %s" % (rank, c))

if __name__ == "__main__":
    do_docking()

