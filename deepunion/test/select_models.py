import mdtraj as mt
import sys
import sklearn
import numpy as np
from sklearn import cluster
import pandas as pd

def load_pdb(fn):
    t = mt.load_pdb(fn)
    top = t.topology
    protein_indices = top.select("protein")
    t = t.atom_slice(protein_indices)

    return t[int(t.n_frames * 0.8): ]

def distance_matrix(dat):
    dist = []
    for x in dat:
        for y in dat:
            d = rmse(x, y)
            dist.append(d)

    return np.array(dist).reshape((dat.shape[0], -1))

def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


if __name__ == "__main__":

    traj_fns = sys.argv[1:]

    traj = None

    for i, fn in enumerate(traj_fns):
        if i == 0:
            traj = load_pdb(fn)
        else:
            traj = traj.join(load_pdb(fn))

    traj = traj.superpose(traj[0])

    xyz = traj.xyz.reshape((traj.n_frames, -1))
    pca = sklearn.decomposition.PCA(n_components=5)
    xyz_t = pca.fit_transform(xyz)

    dist_matrix = distance_matrix(xyz_t)
    dist_matrix = pd.DataFrame(dist_matrix)

    clust = cluster.AgglomerativeClustering(n_clusters=5)
    clust.fit(xyz_t)
    labels = clust.labels_

    for i in range(5):
        # cluster i
        i_label = (labels == i)
        dist = dist_matrix[i_label]
        i_pdb = dist.sum(axis=1).sort_values().index.values[0]
        traj[i_pdb].save_pdb("center_%d.pdb" % i, )


