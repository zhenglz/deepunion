
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols


def metric_fingerprints(metric="Tanimoto"):

    methods = {
        "Tanimoto": DataStructs.TanimotoSimilarity,
        "Dice": DataStructs.DiceSimilarity,
        "Russel": DataStructs.RusselSimilarity,
        "Cosine": DataStructs.CosineSimilarity,
    }

    if metric in methods.keys():
        return methods[metric]
    else:
        print("%s metric is not an option. Will use Tanimoto. "
              % metric)
        return methods["Tanimoto"]


def similarity(mol1, mol2, metric="Tanimoto"):
    """Compare similarity between ligands.

    Parameters
    ----------
    mol1 : str
        The smile code of a molecule
    mol2 : str
        The smile code of a molecule
    metric : str, default = 'Tanimoto'
        The ligand similarity metric. Options:
        Tanimoto, Dice, Russel, Cosine.

    Examples
    --------
    >>> from deepunion import fingerprints
    >>> fingerprints.similarity("CC(C)OCC", "CCOCC")
    0.666666666666666

    References
    ----------
    https://www.rdkit.org/docs/GettingStartedInPython.html

    """
    ms = [Chem.MolFromSmiles(mol1), Chem.MolFromSmiles(mol2)]

    fps = [FingerprintMols.FingerprintMol(x) for x in ms]

    m = metric_fingerprints(metric)

    try:
        return DataStructs.FingerprintSimilarity(fps[0],
                                                 fps[1],
                                                 m)
    except:
        return 0.0

