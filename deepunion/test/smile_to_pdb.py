
from deepunion import builder
import sys, os

if __name__ == "__main__":

    input = sys.argv[1]

    if len(sys.argv) < 2:
        print("Usage: python this_script.py input_smile.list ")
        sys.exit(0)

    out_format = "pdb"

    if not os.path.exists(input):
        print("Input file does not exist. ")
        sys.exit(0)

    compound_dict = {}
    with open(input) as lines:
        for s in lines:
            if len(s.split(",")) and s.split(",")[1] != "":
                compound_dict[s.split(",")[2]] = s.split(",")[1]

    for cid in compound_dict.keys():
        print("Start ", cid, compound_dict[cid])

        if len(compound_dict[cid]):
            try:
                mol = builder.CompoundBuilder(out_format=out_format,
                                              in_format="smile", )
                #print(mol, compound_dict(cid))
                mol.load_mol(compound_dict[cid])
                #print(mol)
                mol.write_mol("mol_%s.%s" % (cid, out_format))
            except:
                print("Fail to generate %s.%s file " % (cid, out_format))

    print("Completed ... ...")


