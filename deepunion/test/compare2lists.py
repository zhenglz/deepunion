from deepunion import fingerprints
import sys
import itertools


if __name__ == "__main__":
    fn1 = sys.argv[1]
    fn2 = sys.argv[2]

    smiles = []
    with open(fn1) as lines:
        smiles.append([x.split(",")[1] for x in lines
                       if (len(x.split(",")) and "#" not in x)])

    with open(fn2) as lines:
        smiles.append([x.split()[0] for x in lines
                   if (len(x.split()) and "#" not in x)])
    #print(smiles)
    smile_pairs = list(itertools.product(smiles[0], smiles[1]))

    similarities = [(fingerprints.similarity(x[0], x[1]), x[0], x[1])
                    for x in smile_pairs if (len(x[0]) and len(x[1]))]

    tofile = open("results.dat", "w")
    for item in similarities:
        if item[0] > 0.7:
            print(item)
            tofile.write("%.3f  %s  %s \n" % (item[0], item[1], item[2]))


