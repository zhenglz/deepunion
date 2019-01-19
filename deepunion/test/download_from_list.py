from deepunion import PubChemDownloader
import sys

if __name__ == "__main__":
    dowloader = PubChemDownloader()

    input = sys.argv[1]
    output = sys.argv[2]

    compounds = []
    with open(input) as lines:
        for s in lines:
            if len(s.split()):
                compounds.append(s.strip("\n"))

    print(compounds)

    tofile = open(output, 'w')

    smiles = {}
    cids = {}
    for c in compounds:
        lig = dowloader.get_compound(c, "name")
        print(c, lig)

        if len(lig):
            smiles[c] = dowloader.get_smile(lig[0])
            cids[c] = dowloader.get_cid(lig[0])
        else:
            smiles[c] = ""
            cids[c] = ""

        tofile.write("%s,%s,%s\n" % (c, smiles[c], cids[c]))

    print("Completed!")

