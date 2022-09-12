"""
Provides classes that supply functions to calculate inverse interatomic distances and their derivatives

Classes:
    Sympy_Grad
    Hard_Code_Grad
"""
from functools import cache, cached_property
import numpy as np
import sympy
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sympy_Grad:
    """
    provides gradients through sympy
    """
    num_atoms: int
    symb: sympy.Array

    @classmethod
    def initialize(cls, natoms):
        """
        initializes the class given the number of atoms
        """
        symb = sympy.Array(sympy.symbols(f"x:{natoms * 3}"))
        return cls(natoms,symb)

    @staticmethod
    def dist(symb,i,j):
        del_x = sympy.Pow((symb[3*i+0]-symb[3*j+0]),2)
        del_y = sympy.Pow((symb[3*i+1]-symb[3*j+1]),2)
        del_z = sympy.Pow((symb[3*i+2]-symb[3*j+2]),2)
        return sympy.sqrt(del_x + del_y + del_z)

    @cached_property
    def z(self):
        pairs = combinations(range(self.num_atoms), 2)
        z = []
        for i, j in pairs:
            z.append(1 / Sympy_Grad.dist(self.symb, i, j))
        return sympy.Array(z)

    @cached_property
    def b(self):
        b = sympy.derive_by_array(self.z,self.symb)
        return sympy.permutedims(b, (1,0))

    @cached_property
    def b2(self):
        b2 = np.array(sympy.derive_by_array(self.b,self.symb))
        return sympy.permutedims(b2, (1,2,0))

    def calc_z(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        """
        return sympy.utilities.lambdify(self.symb, self.z)

    def calc_b(self):
        """
        returns a function that will calculate B given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.3
        """
        #b = sympy.derive_by_array(self.z,self.symb)
        b = self.b
        func = lambda x: np.array(sympy.utilities.lambdify(self.symb, self.b)(*x))
        return func

    def calc_b2(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.4
        """
        b2 = self.b2
        #sympy wants to make this return a list, rather than an np.array
        func = lambda x: np.array(sympy.utilities.lambdify(self.symb, self.b2)(*x))
        return func

@dataclass
class Hard_Code_Grad:
    num_atoms: int

    def calc_z(self):
        pass

    def calc_b(self):
        pass

    def calc_b2(self):
        pass