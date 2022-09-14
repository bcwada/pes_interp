"""
Provides classes that supply functions to calculate inverse interatomic distances and their derivatives

Classes:
    Sympy_Grad
"""
from functools import cache, cached_property
import numpy as np
import sympy
from itertools import combinations
from dataclasses import dataclass


@dataclass
class Sympy_Grad:
    """
    provides gradients through sympy
    """
    num_atoms: int
    symb: sympy.Array
    funcs: dict

    @classmethod
    def initialize(cls, natoms):
        """
        initializes the class given the number of atoms
        """
        symb = sympy.Array(sympy.symbols(f"x:{natoms * 3}"))
        return cls(natoms,symb,{})

    @staticmethod
    def dist(symb,i,j):
        del_x = sympy.Pow((symb[3*i+0]-symb[3*j+0]),2)
        del_y = sympy.Pow((symb[3*i+1]-symb[3*j+1]),2)
        del_z = sympy.Pow((symb[3*i+2]-symb[3*j+2]),2)
        return sympy.sqrt(del_x + del_y + del_z)

    @cached_property
    def inv_dist(self):
        pairs = combinations(range(self.num_atoms), 2)
        z = []
        for i, j in pairs:
            z.append(1 / Sympy_Grad.dist(self.symb, i, j))
        return sympy.Array(z)

    @cached_property
    def b(self):
        b = sympy.derive_by_array(self.inv_dist,self.symb)
        return sympy.permutedims(b, (1,0))

    @cached_property
    def b2(self):
        b2 = np.array(sympy.derive_by_array(self.b,self.symb))
        return sympy.permutedims(b2, (1,2,0))

    @property
    def calc_inv_dist(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        """
        if "calc_z" in self.funcs.keys():
            return self.funcs["calc_z"]
        else:
            func = sympy.utilities.lambdify(self.symb, self.inv_dist)
            unpack = lambda x: np.array(func(*x))
            self.funcs["calc_z"] = unpack
            return unpack

    def calc_b_old(self):
        """
        returns a function that will calculate B given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.3
        """
        if "calc_b" in self.funcs.keys():
            return self.funcs["calc_b"]
        else:
            b = self.b
            func = lambda x: np.array(sympy.utilities.lambdify(self.symb, self.b)(*x))
            self.funcs["calc_b"] = func
            return func

    @property
    def calc_b(self):
        """
        returns a function that will calculate B given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.3
        """
        if "calc_b_alt" in self.funcs.keys():
            return self.funcs["calc_b_alt"]
        else:
            func = sympy.utilities.lambdify(self.symb, self.b)
            unpack = lambda x: np.array(func(*x))
            self.funcs["calc_b_alt"] = unpack
            return unpack

    def calc_b2_old(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.4
        """
        if "calc_b2" in self.funcs.keys():
            return self.funcs["calc_b2"]
        else:
            b2 = self.b2
            #sympy wants to make this return a list, rather than an np.array
            func = lambda x: np.array(sympy.utilities.lambdify(self.symb, self.b2)(*x))
            self.funcs["calc_b2"] = func
            return func

    @property
    def calc_b2(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.4
        """
        if "calc_b2" in self.funcs.keys():
            return self.funcs["calc_b2"]
        else:
            #sympy wants to make this return a list, rather than an np.array
            func = sympy.utilities.lambdify(self.symb, self.b2)
            unpack = lambda x: np.array(func(*x))
            self.funcs["calc_b2"] = unpack
            return unpack
