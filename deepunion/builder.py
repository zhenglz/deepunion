# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem
import os
from subprocess import Popen


class Molecule(object):
    """Molecule parse object with Rdkit.

    Parameters
    ----------
    in_format : str, default = 'smile'
        Input information (file) format.
        Options: smile, pdb, sdf, mol2, mol

    Attributes
    ----------
    molecule_ : rdkit.Chem.Molecule object
    mol_file : str
        The input file name or Smile string
    converter_ : dict, dict of rdkit.Chem.MolFrom** methods
        The file loading method dictionary. The keys are:
        pdb, sdf, mol2, mol, smile


    """

    def __init__(self, in_format="smile"):

        self.format = in_format
        self.molecule_ = None
        self.mol_file = None
        self.converter_ = None
        self.mol_converter()

    def mol_converter(self):
        """The converter methods are stored in a dictionary.

        Returns
        -------
        self : return an instance of itself

        """
        self.converter_ = {
            "pdb": Chem.MolFromPDBFile,
            "mol2": Chem.MolFromMol2File,
            "mol": Chem.MolFromMolFile,
            "smile": Chem.MolFromSmiles,
            "sdf": Chem.MolFromMolBlock,
        }

        #self.converter_ = converter

        return self

    def load_molecule(self, mol_file):
        """Load a molecule to have a rdkit.Chem.Molecule object

        Parameters
        ----------
        mol_file : str
            The input file name or SMILE string

        Returns
        -------
        molecule : rdkit.Chem.Molecule object
            The molecule object

        """

        self.mol_file = mol_file

        if self.format in ["mol2", "mol", "pdb", "sdf"] and\
                os.path.exists(self.mol_file):
            print("Molecule file not exists. ")
            return None

        self.molecule_ = self.converter_[self.format](self.mol_file)

        return self.molecule_


class CompoundBuilder(object):
    """Generate 3D coordinates of compounds.

    Parameters
    ----------
    in_format : str, default='smile'
        The input file format. Options are
        pdb, sdf, mol2, mol and smile.

    out_format : str, default='pdb'
        The output file format. Options are
        pdb, sdf, mol2, mol and smile.
    addH : bool, default = True
        Whether add hydrogen atoms
    optimize : bool, default = True
        Whether optimize the output compound conformer

    Attributes
    ----------
    mol_file : str
        The input file name or smile string.
    molecule_ : rdkit.Chem.Molecule object
        The target compound molecule object
    add_H : bool, default = True

    Examples
    --------
    >>> # generate 3D conformation from a SMILE code and save as a pdb file
    >>> from deepunion import builder
    >>> comp = builder.CompoundBuilder(out_format="pdb", in_format="smile")
    >>> comp.in_format
    'smile'
    >>> comp.load_mol("CCCC")
    <deepunion.builder.CompoundBuilder object at 0x7f0cd1d909b0>
    >>> comp.generate_conformer()
    <deepunion.builder.CompoundBuilder object at 0x7f0cd1d909b0>
    >>> comp.write_mol("mol_CCCC.pdb")
    <deepunion.builder.CompoundBuilder object at 0x7f0cd1d909b0>
    >>> # the molecule has been saved to mol_CCCC.pdb in working directory
    >>> # convert the pdb file into a pdbqt file
    >>> babel_converter("mol_CCCC.pdb", "mol_CCCC.pdbqt")

    """

    def __init__(self, out_format="pdb", in_format="smile",
                 addHs=True, optimize=True):

        self.out_format = out_format
        self.mol_file = None
        self.molecule_ = None
        self.in_format = in_format

        self.add_H = addHs

        self.optimize_ = optimize

        self.converter_ = None
        self.write_converter()

    def generate_conformer(self):
        """Generate 3D conformer for the molecule.

        The hydrogen atoms are added if necessary.
        And the conformer is optimized with a rdkit MMFF
        optimizer

        Returns
        -------
        self : return an instance of itself

        References
        ----------
        Halgren, T. A. “Merck molecular force field. I. Basis, form,
        scope, parameterization, and performance of MMFF94.” J. Comp.
        Chem. 17:490–19 (1996).
        https://www.rdkit.org/docs/GettingStartedInPython.html

        """

        if self.molecule_ is not None:
            # add hydrogen atoms
            if self.add_H:
                self.molecule_ = Chem.AddHs(self.molecule_)

            # generate 3D structure
            AllChem.EmbedMolecule(self.molecule_)

            # optimize molecule structure
            if self.optimize_:
                AllChem.MMFFOptimizeMolecule(self.molecule_)

            return self

        else:
            print("Load molecule first. ")
            return self

    def load_mol(self, mol_file):
        """Load a molecule from a file or a SMILE string

        Parameters
        ----------
        mol_file : str
            The input molecule file name or a SMILE string

        Returns
        -------
        self : return an instance of itself
        """

        self.mol_file = mol_file

        mol = Molecule(in_format=self.in_format)
        self.molecule_ = mol.load_molecule(mol_file)

        return self

    def write_converter(self):
        """Write file methods.

        Returns
        -------
        self : an instance of itself

        """

        converter = {
            "pdb": Chem.MolToPDBFile,
            "sdf": Chem.MolToMolBlock,
            #"mol2": Chem.MolToMol2File,
            "mol": Chem.MolToMolFile,
            "smile": Chem.MolToSmiles,
        }

        self.converter_ = converter

        return self

    def write_mol(self, out_file="compound.pdb"):
        """Write a molecule to a file.

        Parameters
        ----------
        out_file : str, default = compound.pdb
            The output file name.

        Returns
        -------
        self : an instance of itself
        """

        # need to load file first
        self.generate_conformer()

        self.converter_[self.out_format](self.molecule_, out_file)

        return self


def babel_converter(input, output, babelexe="obabel", mode="general"):

    cmd = ""
    if mode == "general":
        cmd = "%s %s -O %s" % (babelexe, input, output)
    elif mode == "AddPolarH":
        cmd = "%s %s -O %s -d" % (babelexe, input, "xxx_temp_noH.pdbqt")
        job = Popen(cmd, shell=True)
        job.communicate()
        cmd = "%s %s -O %s --AddPolarH" % (babelexe, "xxx_temp_noH.pdbqt", output)
    else:
        pass
    job = Popen(cmd, shell=True)
    job.communicate()

    return None


