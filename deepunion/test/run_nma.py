#!/usr/bin/env python

import mdtraj as md
from nma import ANMA
import sys

# Load structure of choice (e.g. Water)
pdb = md.load_pdb(sys.argv[1])

# Initialize ANMA object
anma = ANMA(mode=3, rmsd=0.06, n_steps=50, selection='all')

# Transform the PDB into a short trajectory of a given mode
anma_traj = anma.fit_transform(pdb)

anma_traj.save_pdb(sys.argv[2])

