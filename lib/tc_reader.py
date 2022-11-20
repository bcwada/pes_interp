from dataclasses import dataclass
import re
import numpy as np

# Designed while working on the halothiophene project
# The design is consequently tailored towards that, though hopefully somewhat generalizable
#
# Designed for use on single point energy calculation
# getStates and casciEnergies should work on a geometry optimization, returning ht energies of the last step?
#

@dataclass
class gradient:
    energy: float
    grad: np.array

    @classmethod
    def from_file(cls, path):
        header = re.compile("     dE/dX            dE/dY            dE/dZ")
        energy_re = re.compile("FINAL ENERGY: (-?\d*\.\d*)")
        grad_re = re.compile("(-?\d*\.\d*)\s*(-?\d*\.\d*)\s*(-?\d*\.\d*)")
        found_header = False
        grad = []
        with open(path) as f:
            for line in f:
                if match := energy_re.search(line):
                    energy = float(match[1])
                if found_header:
                    if match := grad_re.search(line):
                        grad.append([float(match[1]),float(match[2]),float(match[3])])
                    else:
                        break
                if header.search(line):
                    found_header = True
        return cls(energy, np.array(grad))

@dataclass
class Hessian:
    hess: np.array

    @classmethod
    def from_bin(cls,path):
        # implementation credit to Nanna
        fh = open(path, 'rb')
        natom = np.fromstring(fh.read(4), dtype=np.int32)[0]
        npoint = np.fromstring(fh.read(4), dtype=np.int32)
        displacement = np.fromstring(fh.read(8), dtype=np.float64)
        G = np.fromstring(fh.read(natom*4*8), dtype=np.float64)
        G = np.reshape(G,(natom,4))
        # geom = []
        # for A in range(natom):
        #     geom.append((
        #         atom_data.atom_symbol_table[int(G[A,3])] if not symbols else symbols[A],
        #         G[A,0],
        #         G[A,1],
        #         G[A,2],
        #         ))  

        hess = np.fromstring(fh.read(9*natom*natom*8), dtype=np.float64)
        hess = np.reshape(hess, (natom*3,natom*3))

        new_hess = cls(hess)
        return new_hess

