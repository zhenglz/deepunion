# -*- coding: utf-8 -*-

from pubchempy import get_compounds


class PubChemDownloader(object):
    """Compounds downloader from PubChem.

    Notes
    -----

    Examples
    --------
    >>> # download progesterone and save it into a pdb file
    >>> from deepunion import downloader
    >>> down = downloader.PubChemDownloader()
    >>> m = down.get_compound("progesterone", type="name")
    >>> smile = down.get_smile(m)
    >>> # now convert SMILE to pdb
    >>> from deepunion import builder
    >>> b = builder.CompoundBuilder("pdb", "smile")
    >>> b.load_mol(smile)
    >>> b.generate_conformer()
    >>> b.write_mol("progesterone.pdb")

    """

    def __init__(self):
        pass

    def get_compound(self, name, type="name"):
        """Get the compound given its name.

        Parameters
        ----------
        name : str
            The general name of a compound
        type : str, default = name
            The name format.

        Returns
        -------
        results : list
            The list of PubchemPy.compound object
        """

        return get_compounds(name, namespace=type)[0]

    def download_from_list(self, name_list, type="name"):
        """Given a list of compounds, download them

        Parameters
        ----------
        name_list
        type

        Returns
        -------

        """

        results = []
        for n in name_list:
            try:
                r = get_compounds(n, namespace=type)
            except:
                print("Ligand %s not found" % n)
                r = []

            if len(r):
                results.append(r[0])
            else:
                results.append(None)

        return results

    def get_smile(self, compound):
        """Return the smile code of a compound

        Parameters
        ----------
        compound : pcp.Compound object

        Returns
        -------
        smile : str
            The smile code of a molecule

        """

        if compound is not None:
            return compound.isomeric_smiles
        else:
            return ""

    def get_cid(self, compound, ):
        """Return the cid of a compound

        Parameters
        ----------
        compound : pubchempy.compound object

        Returns
        -------
        cid : str
            The cid of a compound.
        """
        if compound is not None:
            return compound.cid
        else:
            return ""
