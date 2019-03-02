import numpy as np
import sys, os
import subprocess as sp


class rewritePDB(object):
    """
    Modify pdb file by changing atom indexing, resname, res sequence number and chain id

    Parameters
    ----------

    Attributes
    ----------

    """
    def __init__(self, inpdb):
        self.pdb = inpdb

    def pdbRewrite(self, input, output, chain, atomStartNdx, resStartNdx):
        """
        change atom id, residue id and chain id
        :param input: str, input pdb file
        :param output: str, output file name
        :param chain: str, chain id
        :param atomStartNdx: int,
        :param resStartNdx: int
        :return:
        """
        resseq = int(resStartNdx)
        atomseq = int(atomStartNdx)
        chainname = chain

        newfile = open(output,'w')
        resseq_list = []

        try :
            with open(input) as lines :
                for s in lines :
                    if "ATOM" in s and len(s.split()) > 6 :
                        atomseq += 1
                        newline = s
                        newline = self.atomSeqChanger(newline, atomseq)
                        newline = self.chainIDChanger(newline, chainname)
                        if len(resseq_list) == 0 :
                            newline = self.resSeqChanger(newline, resseq)
                            resseq_list.append(int(s[22:26].strip()))
                        else :
                            if resseq_list[-1] == int(s[22:26].strip()) :
                                newline = self.resSeqChanger(newline, resseq)
                            else :
                                resseq += 1
                                newline = self.resSeqChanger(newline, resseq)
                            resseq_list.append(int(s[22:26].strip()))
                        newfile.write(newline)
                    else :
                        newfile.write(s)
        except FileExistsError :
            print("File %s not exist" % self.pdb)

        newfile.close()
        return 1

    def resSeqChanger(self, inline, resseq):
        resseqstring = " "*(4 - len(str(resseq)))+str(resseq)
        newline = inline[:22] + resseqstring + inline[26:]
        return newline

    def atomSeqChanger(self, inline, atomseq):
        atomseqstring = " " * (5 - len(str(atomseq))) + str(atomseq)
        newline = inline[:6] + atomseqstring + inline[11:]
        return newline

    def resNameChanger(self, inline, resname):
        resnamestr = " " * (4 - len(str(resname))) + str(resname)
        newline = inline[:16] + resnamestr + inline[20:]
        return newline

    def chainIDChanger(self, inline, chainid) :
        newline = inline[:21] + str(chainid) + inline[22:]
        return newline

    def atomNameChanger(self, inline, new_atom_name):
        newline = inline[:12] + "%4s" % new_atom_name + inline[16:]
        return newline

    def combinePDBFromLines(self, output, lines):
        """
        combine a list of lines to a pdb file

        Parameters
        ----------
        output
        lines

        Returns
        -------

        """

        with open(output, "wb") as tofile :
            tmp = map(lambda x: tofile.write(x), lines)

        return 1

    def swampPDB(self, input, atomseq_pdb, out_pdb, chain="B"):
        """
        given a pdb file (with coordinates in a protein pocket), but with wrong atom
        sequence order, try to re-order the pdb for amber topology building

        Parameters
        ----------
        input:str,
            the pdb file with the correct coordinates
        atomseq_pdb:str,
            the pdb file with correct atom sequences
        out_pdb: str,
            output pdb file name
        chain: str, default is B
            the chain identifier of a molecule

        Returns
        -------

        """

        tofile = open("temp.pdb", 'w')

        crd_list = {}

        ln_target, ln_source = 0, 0
        # generate a dict { atomname: pdbline}
        with open(input) as lines :
            for s in [x for x in lines if ("ATOM" in x or "HETATM" in x)]:
                crd_list[s.split()[2]] = s
                ln_source += 1

        # reorder the crd_pdb pdblines, according to the atomseq_pdb lines
        with open(atomseq_pdb) as lines:
            for s in [x for x in lines if ("ATOM" in x or "HETATM" in x)]:
                newline = crd_list[s.split()[2]]
                tofile.write(newline)
                ln_target += 1

        tofile.close()

        if ln_source != ln_target:
            print("Error: Number of lines in source and target pdb files are not equal. (%s %s)"%(input, atomseq_pdb))

        # re-sequence the atom index
        self.pdbRewrite(input="temp.pdb", atomStartNdx=1, chain=chain, output=out_pdb, resStartNdx=1)

        os.remove("temp.pdb")

        return None


class coordinatesPDB(object):
    def __init__(self):
        pass

    def replaceCrdInPdbLine(self, line, newxyz):
        """Input a line of pdb file, and a new vector of xyz values,
        the old xyz values will be replaces, and return a new pdb line

        Parameters
        ----------
        line : str
            A line from pdb file
        newxyz : list, shape = [3, ]
            The new xyz coordinates, in unit nanometer

        Returns
        -------
        new_line : str
            The new PDB line with new xyz coordinates

        """

        if "ATOM" in line or "HETATM" in line:
            head = line[:30]
            tail = line[54:]

            newline = head + "{0:8.3f}{1:8.3f}{2:8.3f}".format(newxyz[0], newxyz[1], newxyz[2]) + tail

        else :
            print("WARNING: %s is not a coordination line"%line)
            newline = ""

        return newline

    def getAtomCrdFromLines(self, lines):
        """Given a list of atom pbd lines, return their coordinates in a 2d list

        Parameters
        ----------
        lines : list
            A list of pdb lines contains coordinates

        Returns
        -------
        coordinates : np.ndarray, shape = [ N, 3]
            The coordinates of selected pdb lines, N is the number of
            lines.
        """

        atomCrd = list(map(lambda x: [float(x[30:38].strip()),float(x[38:46].strip()),
                                      float(x[46:54].strip())],lines))

        return np.array(atomCrd)

    def getAtomCrdByNdx(self, singleFramePDB, atomNdx=['1',]):
        """Input a pdb file and the atom index, return the crd of the atoms

        Parameters
        ----------
        singleFramePDB : str
            Input pdb file name.
        atomNdx : list
            A lis of atoms for coordinates extraction.

        Returns
        -------
        coordinates : list, shape = [ N, 3]
            The coordinates for the selected lines. N is the number of
            selected atom indices.

        """

        atomCrd = []
        with open(singleFramePDB) as lines :
            lines = [s for s in lines if len(s) > 4 and
                     s[:4] in ["ATOM","HETA"] and
                     s.split()[1] in atomNdx]
            atomCrd = map(lambda x: [float(x[30:38].strip()),
                                     float(x[38:46].strip()),
                                     float(x[46:54].strip())],
                          lines)
        return list(atomCrd)


def void_area(centriod, diameter):
    """

    Parameters
    ----------
    centriod : list, or iterable
        The center for a void area
    diameter : float

    Returns
    -------

    """

    area = []
    for i in range(len(centriod)):
       area.append([centriod[i] - 0.5 * diameter, centriod[i] + 0.5 * diameter])

    return area


def dot_in_range(dot, xy_range):

    return dot >= xy_range[0] and dot < xy_range[1]


def point_in_area(area, point):

    in_area = []
    for i in range(len(point)):
        in_area.append(dot_in_range(point[i], area[i]))

    return all(in_area)


def sliding_box_centers(center, step_size=0.3, nstep=3, along_xyz=[True, True, True]):

    centriods = []

    for i in range(-1*nstep, nstep+1):
        for j in range(-1*nstep, nstep+1):
            for k in range(-1*nstep, nstep+1):
                c = center.copy()
                for m in range(3):
                    c[m] = center[m] + step_size * [i, j, k][m]
                centriods.append(c)

    return centriods


def atoms_in_boxs(fn, atom_ndx, box_center, box_diameter):
    pdbio = coordinatesPDB()
    area = void_area(box_center, box_diameter)

    coords = pdbio.getAtomCrdByNdx(fn, atom_ndx)
    in_void_area = [point_in_area(area=area, point=x) for x in coords]

    return zip(atom_ndx, in_void_area)


def get_atom_ndx(fn, atom_names=['CA'], lig_code="LIG"):
    with open(fn) as lines:
        # only consider protein atoms
        lines = [ x for x in lines if ("ATOM" in x and lig_code not in x and x.split()[2] in atom_names)]
        atom_ndx = [x.split()[1] for x in lines if len(x.split()) > 2]

    return atom_ndx


def get_resid(fn, atom_names=['CA'], lig_code="LIG"):
    with open(fn) as lines:
        # only consider protein atoms
        lines = [ x for x in lines if ("ATOM" in x and lig_code not in x and x.split()[2] in atom_names)]
        resid = [x[22:26].strip()+"_"+x[21] for x in lines if len(x.split()) > 2]

    return resid


def resid_in_box(box_center, diameter, fn, atom_names=['CA'], lig_code="LIG"):

    atom_ndx = get_atom_ndx(fn, atom_names, lig_code)
    resid = get_resid(fn, atom_names, lig_code)

    resid_atomndx = dict(zip(atom_ndx, resid))

    resid_is_inbox = atoms_in_boxs(fn, atom_ndx, box_center, diameter)

    selected_resid = []

    for item in resid_is_inbox:
        if item[1]:
            selected_resid.append(resid_atomndx[item[0]])

    return selected_resid


def trim_sidechain(fn, out_fn, resid_list, atoms_to_keep=['CA', 'N', 'O', 'C'], mutate_to="ALA", chains="A"):

    rpdb = rewritePDB(fn)

    tofile = open(out_fn, 'w')

    with open(fn) as lines:
        for s in lines:
            if "ATOM" in s and s[22:26].strip() in resid_list and s[21] in chains:
                if s.split()[2] not in atoms_to_keep:
                    pass
                else:
                    new_line = rpdb.resNameChanger(s, mutate_to)
                    tofile.write(new_line)
            elif "ATOM" in s or "HETATM" in s or "TER" in s:
                tofile.write(s)

    return tofile


def origin_to_zero(fn, out_fn, origin):

    with open(fn) as lines:
        lines = [x for x in lines if (x.split()[0] in ["ATOM", "HETATM"]
                                      and len(x.split())> 3)]
        plines = lines.copy()

    vector = -1.0 * np.array(origin)

    pio = coordinatesPDB()

    tofile = open(out_fn, 'w')

    for s in plines:
        crds = pio.getAtomCrdFromLines([s, ])[0]
        new_crds = np.array(crds) + vector
        nl = pio.replaceCrdInPdbLine(s, new_crds)

        tofile.write(nl)
    tofile.close()


def run_tleap_tofix(in_pdb, out_pdb):

    leapin = """
    source leaprc.ff14SB
    pdb = loadpdb %s
    check pdb
    savepdb pdb %s
    """ % (in_pdb, out_pdb)

    tofile = open("LeapIn.in", 'w')
    tofile.write(leapin)

    job = sp.Popen("tleap -f LeapIn.in", shell=True)
    job.communicate()

    return 1


def main():
    # how to use:
    # python region_mutate.py 2ovh_clean.pdb

    if len(sys.argv) < 2:
        print("Usage: \npython region_mutation.py input.pdb")

    in_pdb = sys.argv[1]

    # the sliding stride, =0.3 nm as stated in the paper
    step_size = 0.3
    # how many moves in each direction. for example, if you give
    # 3, that means, you will move 3 times in -x and x directions
    # together you will generate 3 + 3 + 1 = 7 structures. You may not
    # need to change it here. As the paper use 7 boxes per dimension.
    nsteps_to_slide = 3

    # set a center point here, give the x, y, z coordinate
    # of the center point, in unit nanometer
    center = [29.582,  -3.843,  26.456]
    # the size of the void box, you may need to modify it. The paper use
    # 10.0 nm
    diameter = 20.0
    # the chain identifier
    chains = ["A", "B"]

    origin = np.array(center) - 0.5 * diameter
    origin_to_zero(in_pdb, "temp.pdb", origin)

    # for x, y, z direction
    dimensions = ["x", "y", "z"]

    centroids = sliding_box_centers(center, step_size,
                                    nsteps_to_slide,
                                    [True, True, True])
    #print(centroids)
    for i, c in enumerate(centroids):
        print(c, i)
        resids = resid_in_box(c, diameter, in_pdb, 'CA', 'LIG')
        # print(resids)
        tofile = trim_sidechain("temp.pdb", "t1.pdb", resids, chains=chains)
        tofile.close()

        #run_tleap_tofix("t1.pdb", "t2.pdb")
        tofile = open("out_%d.pdb" % i, "w")
        with open("t1.pdb") as lines:
            tofile.write("CRYST1  %7.3f  %7.3f  %7.3f  90.00  90.00  90.00               1 \n" %
                         (diameter, diameter, diameter))
            for s in lines:
                tofile.write(s)
            tofile.close()

