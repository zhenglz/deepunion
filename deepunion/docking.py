
import subprocess as sp
import numpy as np
import mdtraj as mt
from rdkit import Chem
from .region_mutate import coordinatesPDB

"""
Docking Routine

1. perform docking with vina
2. output parse
3. docking energy recording

"""


class VinaDocking(object):

    def __init__(self, vina_exe="vina"):
        self.vina_exe = vina_exe

    def vina_config(self, receptor, ligand, outname,
                    n_cpus, exhaustiveness, center, boxsize):

        with open("vina.config", "w") as tofile:

            tofile.write("receptor = %s \n" % receptor)
            tofile.write("ligand = %s \n" % ligand)
            tofile.write("output = %s \n" % outname)

            # center of x y z
            tofile.write("center_x = %.3f \n" % center[0])
            tofile.write("center_y = %.3f \n" % center[1])
            tofile.write("center_z = %.3f \n" % center[2])
            # box size of x y z
            tofile.write("size_x = %.2f \n" % boxsize[0])
            tofile.write("size_y = %.2f \n" % boxsize[1])
            tofile.write("size_z = %.2f \n" % boxsize[2])

            tofile.write("ncpus = %d \n" % n_cpus)
            tofile.write("exhaustiveness = %d \n" % exhaustiveness)

        return "vina.config"

    def run_docking(self, receptor, ligand, outname,
                    n_cpus, exhaustiveness, center, boxsize):

        config = self.vina_config(receptor, ligand, outname,
                                  n_cpus, exhaustiveness,
                                  center, boxsize)

        job = sp.Popen("vina --config %s " % config)
        job.communicate()

        job.terminate()

        return self


class ReceptorPrepare(object):

    def __init__(self, receptor):

        self.receptor = receptor

    def pocket_center(self, LIG="", res_sele="all"):
        if len(LIG):
            with open(self.receptor) as lines:
                lig_lines = [x for x in lines if (LIG in x and len(x.split())
                                                  and x.split()[0] in
                                                  ["ATOM", "HETATM"])]

            # read coordinates
            coord = coordinatesPDB().getAtomCrdFromLines(lig_lines)
            coord = np.array(coord)

        else:
            ref = mt.load(self.receptor)

            # xyz
            coord = np.array([])

        return np.mean(coord, axis=0)

    def receptor_addH(self, out_pdb="temp.pdb"):

        mol = Chem.MolFromPDBFile(self.receptor)

        Chem.AddHs(mol)

        Chem.MolToPDBFile(mol)

        return self

    def pdb2pdbqt(self, keep_polarH=True):
        pass


