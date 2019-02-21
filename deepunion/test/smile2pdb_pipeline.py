
from deepunion import builder
import sys, os

if __name__ == "__main__":

    input = sys.argv[1]

    if len(sys.argv) < 2:
        print("Usage: python this_script.py input_smile.list ")
        sys.exit(0)

    out_format = "pdb"
    i = 0
    with open(input) as lines:
        for s in lines:
            n = s.split()[1]
            sm = s.split()[0]

            if i % 1000 == 0:
                print("Progress %d out 4900 k" % i)

            try:
                #print(sm, i)
                mol = builder.CompoundBuilder(out_format=out_format,
                                              in_format="smile", )

                mol.load_mol(sm)

                mol.write_mol("%s.%s" % (n, out_format))
                builder.babel_converter("%s.%s" % (n, out_format),
                                        "%s.pdbqt" % n, mode='AddPolarH')
            except:
                print("Fail to generate %d.%s file " % (i, out_format))
            i += 1
