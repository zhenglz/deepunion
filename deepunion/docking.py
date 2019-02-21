
import subprocess as sp
import numpy as np
import mdtraj as mt
from rdkit import Chem
from deepunion.region_mutate import coordinatesPDB
from deepunion import builder
import argparse


"""
Docking Routine

1. perform docking with vina
2. output parse
3. docking energy recording

"""


class VinaDocking(object):

    def __init__(self, vina_exe="vina"):
        self.vina_exe = vina_exe

        self.config = None

    def vina_config(self, receptor, ligand, outname,
                    n_cpus, exhaustiveness, center,
                    boxsize=[30, 30, 30], logfile="log.log", n_modes=1):

        with open("vina.config", "w") as tofile:

            tofile.write("receptor = %s \n" % receptor)
            tofile.write("ligand = %s \n" % ligand)
            tofile.write("out = %s \n" % outname)

            # center of x y z
            tofile.write("center_x = %.3f \n" % center[0])
            tofile.write("center_y = %.3f \n" % center[1])
            tofile.write("center_z = %.3f \n" % center[2])
            # box size of x y z
            tofile.write("size_x = %.2f \n" % boxsize[0])
            tofile.write("size_y = %.2f \n" % boxsize[1])
            tofile.write("size_z = %.2f \n" % boxsize[2])

            tofile.write("cpu = %d \n" % n_cpus)
            tofile.write("num_modes = %d \n" % n_modes)
            tofile.write("exhaustiveness = %d \n" % exhaustiveness)

            tofile.write("log = %s \n" % logfile)

        self.config = "vina.config"

        return self

    def run_docking(self):

        if self.config is not None:

            job = sp.Popen("vina --config %s " % self.config, shell=True)
            job.communicate()

            job.terminate()
        else:
            print("Please setup config first")

        return self


class ReceptorPrepare(object):

    def __init__(self, receptor):

        self.receptor = receptor

    def pocket_center(self, LIG="", res_sele="all"):
        if len(LIG):
            with open(LIG) as lines:
                lig_lines = [x for x in lines if x.split()[0] in ["ATOM", "HETATM"]]

            # read coordinates
            coord = coordinatesPDB().getAtomCrdFromLines(lig_lines)
            coord = np.array(coord)

        else:
            # TODO: mdtraj extract specific coordinates
            ref = mt.load(self.receptor)
            sele = ref.topology.select(res_sele)
            # xyz of first frame
            coord = ref.xyz[0][sele]

        return np.mean(coord, axis=0)

    def receptor_addH(self, out_pdb="temp.pdb"):

        mol = Chem.MolFromPDBFile(self.receptor)

        Chem.AddHs(mol)

        Chem.MolToPDBFile(mol, out_pdb)

        return self


def rmsd(mol1, mol2):
    m1 = mt.load(mol1).xyz[0]
    m2 = mt.load(mol2).xyz[0]

    rmsd = np.sum((m1 - m2).ravel() ** 2 / m1.shape[0])

    return np.sqrt(rmsd)


def pdb2pdbqt(inp, out, keep_polarH=True):

    if keep_polarH:
        mode = "AddPolarH"
    else:
        mode = "general"
    print(mode)
    builder.babel_converter(inp, out, "obabel")

    return None


def main():

    d = """
    Perform molecular docking using AutoDock Vina.
    """

    parser = argparse.ArgumentParser(description=d)

    parser.add_argument("-rec", type=str, default="receptor.pdbqt",
                        help="Input. Default is receptor.pdbqt. \n"
                             "The input receptor conformation.")
    parser.add_argument("-lig", type=str, default="ligand.pdbqt",
                        help="Input. Default is ligand.pdbqt. "
                             "The input ligand conformation.")
    parser.add_argument("-out", type=str, default="output_",
                        help="Output. Optional. Default is output_"
                             "The prefix of the output")
    parser.add_argument("-cal_center", type=int, default=1,
                        help="Input, optional. Whether calculate the binding pocket"
                             "automately.")

    args = parser.parse_args()
    # if prepare ligand
    rec = args.rec
    lig = args.lig

    #babel_converter(lig, lig+".pdb")
    pdb2pdbqt(lig, lig+".pdbqt", )
    pdb2pdbqt(lig, lig+".pdb", keep_polarH=False)

    rec_prep = ReceptorPrepare(rec)
    #rec_prep.receptor_addH("H_"+rec)
    xyz_c = rec_prep.pocket_center(LIG=lig+".pdb")
    print(xyz_c)
    pdb2pdbqt(rec, "temp.pdbqt")
    job = sp.Popen("awk '$1 ~ /ATOM/ {print $0}' temp.pdbqt > %s.pdbqt"%rec, shell=True)
    job.communicate()

    docking = VinaDocking()
    docking.vina_config(rec+".pdbqt", lig+".pdbqt", args.out, 16, 32, xyz_c, [40, 40, 40], "log_vina.log", n_modes=20)
    docking.run_docking()

#if "__name__" == __main__:
    main()

