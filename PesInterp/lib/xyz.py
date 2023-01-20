import re
import numpy as np
import os.path
from dataclasses import dataclass
from typing import List
from math import cos, sin
from scipy.spatial.transform import Rotation as R

def rodrigues(pos,axis,angle):
    """Rotates a point POS around AXIS by ANGLE and returns.

    POS and AXIS must be numpy arrays of length 3.
    AXIS must be a nonzero vector.
    ANGLE is in radians."""

    if (np.linalg.norm(axis)<0.001):
        raise Exception("Axis not normalizable!")

    axis=axis/np.linalg.norm(axis)

    pos_proj=np.dot(axis,pos)*axis 
    pos_orth=np.cross(axis,pos)

    return pos*cos(angle) + pos_orth*sin(angle) + pos_proj*(1-cos(angle))


@dataclass
class Geometry:
    numAtoms: int
    comment: str
    atoms: np.array
    coords: np.array

    @classmethod
    def from_file(cls, xyzfile):
        with open(xyzfile) as f:
            numAtoms = int(f.readline())
            comment = f.readline().strip()
            atoms = []
            coords = np.zeros((numAtoms, 3))
            parser = re.compile("\s*([A-Za-z]*)" + ("\s*(-?[0-9]*\.[0-9]*)" * 3))
            for i in range(numAtoms):
                entries = parser.findall(f.readline())[0]
                atoms.append(entries[0])
                coords[i, 0] = float(entries[1])
                coords[i, 1] = float(entries[2])
                coords[i, 2] = float(entries[3])
        return cls(numAtoms, comment, np.array(atoms), coords)

    def write_file(self, filename):
        if os.path.exists(filename):
            raise Exception("Can't write file, file already exists")
        with open(filename, "w") as f:
            self._write_into_file(f)

    def _write_into_file(self, f):
            f.write(str(self.numAtoms) + "\n")
            f.write(self.comment + "\n")
            tenSpaces = " " * 7
            for i in range(self.numAtoms):
                element = self.atoms[i]
                if len(element) == 1:
                    element = element + "  "
                else:
                    element = element + " "
                f.write(f"{element}{tenSpaces}{self.coords[i][0]:13.10f}{tenSpaces}{self.coords[i][1]:13.10f}{tenSpaces}{self.coords[i][2]:13.10f}\n")

    @property
    def num_atoms(self):
        return self.numAtoms

    def rot_bond_rodrigues(self,rot_atoms,a1,a2,angle):
        """Rotate ROT_ATOMS along A1-A2 axis by ANGLE.

        ROT_ATOMS is an array of index values of atoms to rotate.
        A1, A2 are atom index values defining the axis of rotation.
        ANGLE is in radians.
        """
        axis=self.coords[a2] - self.coords[a1]

        for i in rot_atoms:
            self.coords[i] = rodrigues(self.coords[i],axis,angle)

    def bond_rot(self, angle, bond_atoms, rot_atoms):
        #rotate all the atoms given as indices in rot_atoms counterclockwise around the vector from bond_atoms[0] to bond_atoms[1]
        shift_vec = self.coords[bond_atoms[0]]
        self.coords = self.coords-shift_vec
        rot_vec = self.coords[bond_atoms[1]]/np.linalg.norm(self.coords[bond_atoms[1]])
        rot_vec = rot_vec*np.sin(angle/2)
        r = R.from_quat([rot_vec[0],rot_vec[1],rot_vec[2],np.cos(angle/2)])
        for i in rot_atoms:
            self.coords[i] = r.apply(self.coords[i])
        self.coords = self.coords+shift_vec

    def dist(self, key1, key2):
        """returns the distance between the atoms at indices key1 and key2"""
        return np.sqrt(np.sum(np.square(self.coords[key1] - self.coords[key2])))

@dataclass
class combinedGeoms:
    geometries: List[Geometry]

    @classmethod
    def readFile(cls, xyzfile):
        geometries = []
        with open(xyzfile) as f:
            lines = f.readlines()
        numAtomsRegex = re.compile("^\s*[0-9]+$")
        coordRegex = re.compile("\s*([A-Za-z]*)" + ("\s*(-?[0-9]*\.[0-9]*)" * 3))
        nullRegex = re.compile("^\s+$")
        count = 0
        while count < len(lines):
            if nullRegex.match(lines[count]):
                count += 1
            elif numAtomsRegex.match(lines[count]):
                numAtoms = int(lines[count])
                comment = lines[count + 1].strip()
                count += 2
                atoms = []
                coords = np.zeros((numAtoms, 3))
                for i in range(numAtoms):
                    entries = coordRegex.findall(lines[count])[0]
                    atoms.append(entries[0])
                    coords[i, 0] = float(entries[1])
                    coords[i, 1] = float(entries[2])
                    coords[i, 2] = float(entries[3])
                    count += 1
                geometries.append(Geometry(numAtoms, comment, atoms, coords))
        return cls(geometries)

    def __getitem__(self, key):
        return self.geometries[key]

    def write_last_geom(self, file):
        self.geometries[-1].write_file(file)

    def get_last_geom(self):
        return self.geometries[-1]

    def write_all(self,filename):
        #TODO put the filewriting in a shared function, this is copy of write function for single geometry
        if os.path.exists(filename):
            raise Exception("Can't write file, file already exists")
        with open(filename, "w") as f:
            for g in self.geometries:
                g._write_into_file(f)
