import mdtraj as mt
from deepunion.docking import coordinatesPDB
import numpy as np
from prody import *
import sys


class EssentialDynamics(object):

    def __init__(self):
        pass

    def transform_xyz(self, xyz, vectors, delta):
        """transform xyz along a vector
        eg. increase movements of an atom along the PC1 vectors

        Parameters
        ----------
        xyz : list,
            xyz value
        vectors : list,
            vectors for xyz
        delta : float,
            stride, unit nano meter

        Returns
        -------
        newxyz : list
            new list of xyz values

        """

        newxyz = list(map(lambda x, y: x + y*delta, xyz, vectors))

        return newxyz

    def pdbIncreaseMotion(self, pdbin, vectors, delta=0.5):
        """Increase motions of a pdb given its PC component eigenvectors

        Parameters
        ----------
        pdbin
        vectors: ndarray, shape=[M, 3]
            a M * 3 matrix, M means num of Ca atoms or heavy atoms
        delta: float, default=0.5
            the stride for atom movement

        Returns
        -------
        newxyzs : np.ndarray, shape = [ M, 3]
            the new xyz coordinates, M is number of atoms
        newlines : list, length = M
            the new pdb lines, M is number of atoms

        """

        pdbio = coordinatesPDB()

        with open(pdbin) as lines:
            lines = [x for x in lines if ("ATOM" in x or "HETATM" in x)]

            coords = pdbio.getAtomCrdFromLines(lines)

            newxyzs = []
            newlines = []

            if vectors.shape[0] == len(coords):
                for i in range(len(coords)):
                    newxyz = self.transform_xyz(coords[i], list(vectors[i]), delta)
                    newxyzs.append(newxyz)
                    newlines.append(pdbio.replaceCrdInPdbLine(lines[i], newxyz))

        return newxyzs, newlines

    def genEDA_essemble(self, pdbin, pdbout, vector, no_files=20, delta=0.5, numres=250):
        """Generate an essemble of pdb files to increase the PC motions

        Parameters
        ----------
        pdbin : str
            Input pdb file name
        pdbout : str
            The output EDA ensemble pdb file name
        vector : np.ndarray
            The input eigenvectors
        no_files : int, default = 20
            The number of pdb frames in the output ensemble
        delta : float, default = 0.5
            The stride size.
        numres : int

        Returns
        -------
        self : the instance itself
        """

        PI = 3.14159

        with open(pdbout, 'w') as tofile:
            for i in range(no_files):
                length = delta * np.cos(2.0 * PI * (float(i) / float(no_files)) - PI)
                print(length)
                tofile.write("MODEL   %d \n" % i)
                t, nlines = self.pdbIncreaseMotion(pdbin, vector, delta=length)
                for x in nlines:
                    tofile.write(x)
                tofile.write("ENDMDL  \n")

        return self


class NMA(object):
    def __init__(self, n_modes=3):

        self.eigvectors = np.array([])
        self.n_modes = 3

    def GNM(self, pdb_fn="1a28.pdb"):

        protein = parsePDB(pdb_fn)
        gnm = GNM('protein-ligand')

        selected = protein.select('all')

        gnm.buildKirchhoff(selected)
        gnm.calcModes()

        eigen_vectors = []

        for i in range(self.n_modes):
            slowest_mode = gnm[i]
            # append eigenvectors to the list
            eigen_vectors.append(slowest_mode.getEigvec())

        self.eigvectors = eigen_vectors

        return self.eigvectors

    def write_traj(self, pdb_fn, out_fn="gnm_traj_mode0.pdb",
                   eigvec=np.array([]), n_frames=20, step_size=0.1):

        eda = EssentialDynamics()

        eda.genEDA_essemble(pdb_fn, out_fn, vector=eigvec,
                            no_files=n_frames, delta=step_size)

        return self

def main():

    inpdb = sys.argv[1]
    outpdb = sys.argv[2]

    gnm = NMA()

    gnm.GNM(inpdb)
    ev = np.reshape(gnm.eigvectors[0], [-1, 3])

    print(ev)

    gnm.write_traj(inpdb, outpdb, ev, 10, step_size=0.1)

main()

