# -*- coding: utf-8 -*-

import pubchempy as pcp
from pubchempy import get_compounds


class PubChemDownloader(object):

    def __init__(self):
        pass

    def get_compound(self, name, type="name"):

        return get_compounds(name, namespace=type)[0]

    def download_from_list(self, name_list, type="name"):

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

    def get_smile(self, compound, ):
        """Return the smile code of a compound

        Parameters
        ----------
        compound : pcp.Compound object

        Returns
        -------
        smile : str
            The smile code of a molecule

        """

        return compound.isomeric_smiles

    def get_cid(self, compound, ):

        return compound.cid
