from deepunion import downloader
import sys

if __name__ == "__main__":
    inp = sys.argv[1]

    with open(inp) as lines:
        ids = [x.split()[0] for x in lines]

    zinc = downloader.ZINCDownloader()
    smiles = zinc.get_by_ids(ids, 2)

    with open(sys.argv[2], 'w') as tofile:
        for s, id in zip(smiles, ids):
            tofile.write("%s  %s \n"%(s, id))

    print("Completed")

