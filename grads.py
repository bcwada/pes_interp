"""
Provides classes that supply functions to calculate inverse interatomic distances and their derivatives

Classes:
    Sympy_Grad
"""
from functools import cached_property
import numpy as np
import sympy
from itertools import combinations
from dataclasses import dataclass


@dataclass
class Sympy_Grad:
    """
    Derive functions for inverse distances, gradients and hessian from computer algebra.
    """
    num_atoms: int
    symb: sympy.Array
    funcs: dict

    def __init__(self, natoms):
        symb = sympy.Array(sympy.symbols(f"x:{natoms * 3}"))
        self.num_atoms =  natoms
        self.symb = symb
        self.funcs = {}

    # @classmethod
    # def initialize(cls, natoms):
    #     """
    #     Initialize the class for a fixed number of atoms NATOMS.

    #     This initializes an array of symbols used for sympy's computer
    #     algebra.
    #     """
    #     symb = sympy.Array(sympy.symbols(f"x:{natoms * 3}"))
    #     return cls(natoms,symb,{})

    @staticmethod
    def dist(symb,i,j):
        """
        Generate symbolic distance between two atoms I and J.
        """
        del_x = sympy.Pow((symb[3*i+0]-symb[3*j+0]),2)
        del_y = sympy.Pow((symb[3*i+1]-symb[3*j+1]),2)
        del_z = sympy.Pow((symb[3*i+2]-symb[3*j+2]),2)
        return sympy.sqrt(del_x + del_y + del_z)

    @cached_property
    def inv_dist(self):
        """
        Generate symbolic inverse distances.
        """
        pairs = combinations(range(self.num_atoms), 2)
        z = []
        for i, j in pairs:
            z.append(1 / Sympy_Grad.dist(self.symb, i, j))
        return sympy.Array(z)

    @cached_property
    def inv_jacobian(self):
        """
        Generate symbolic Jacobian of inverse distances w.r.t. cartesian coordinates.
        """
        J = sympy.derive_by_array(self.inv_dist,self.symb)
        return sympy.permutedims(J, (1,0))

    @cached_property
    def inv_hessian(self):
        """
        Generate symbolic Hessian of inverse distances w.r.t. cartesian coordinates.
        """
        H = np.array(sympy.derive_by_array(self.inv_jacobian,self.symb))
        return sympy.permutedims(H, (1,2,0))

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
            b = self.inv_jacobian
            func = lambda x: np.array(sympy.utilities.lambdify(self.symb, self.inv_jacobian)(*x))
            self.funcs["calc_b"] = func
            return func

    @property
    def calc_inv_jacobian(self):
        """
        returns a function that will calculate B given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.3
        """
        if "calc_b_alt" in self.funcs.keys():
            return self.funcs["calc_b_alt"]
        else:
            func = sympy.utilities.lambdify(self.symb, self.inv_jacobian)
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
            b2 = self.inv_hessian
            #sympy wants to make this return a list, rather than an np.array
            func = lambda x: np.array(sympy.utilities.lambdify(self.symb, self.inv_hessian)(*x))
            self.funcs["calc_b2"] = func
            return func

    @property
    def calc_inv_hessian(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.4
        """
        if "calc_b2" in self.funcs.keys():
            return self.funcs["calc_b2"]
        else:
            #sympy wants to make this return a list, rather than an np.array
            func = sympy.utilities.lambdify(self.symb, self.inv_hessian)
            unpack = lambda x: np.array(func(*x))
            self.funcs["calc_b2"] = unpack
            return unpack

@dataclass
class Exact_Grad:
    """
    Functions for inverse distances, gradients and hessian
    Should be identical to Sympy_Grad except faster to initialize
    (with one small caviat, that the returned functions here are tied
    to the object while Sympy_Grad actually returns a new function)
    """
    num_atoms: int

    @property
    def calc_inv_dist(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        """
        def inv_dist(one_d_coords):
            num_dists = (self.num_atoms*(self.num_atoms-1))//2
            z = np.zeros(num_dists)
            z_ind = 0
            for i in range(self.num_atoms):
                for j in range(i+1,self.num_atoms):
                    delx = one_d_coords[3*i+0] - one_d_coords[3*j+0]
                    dely = one_d_coords[3*i+1] - one_d_coords[3*j+1]
                    delz = one_d_coords[3*i+2] - one_d_coords[3*j+2]
                    dist = np.sqrt(np.power(delx,2)+np.power(dely,2)+np.power(delz,2))
                    z[z_ind] = dist
                    z_ind += 1
            return 1/z
        return inv_dist
    
    @property
    def calc_inv_jacobian(self):
        """
        returns a function that will calculate B given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.3
        """
        def inv_jacobian(one_d_coords):
            pass
        return inv_jacobian

    @property
    def calc_inv_hessian(self):
        """
        returns a function that will calculate z given the 1D coords (atom1x,atom1y,atom1z,atom2x...)
        eq. 3.4
        """
        def inv_hessian(one_d_coords):
            pass
        return inv_hessian

@dataclass
class numerical_grad:

    @staticmethod
    def grad_2pt(func, coords, delta=0.01):
        """
        take coords in as a 1D array
        """
        L = []
        num_coords = len(coords)
        for i in range(num_coords):
            pcoords = coords.copy()
            mcoords = coords.copy()
            pcoords[i] += delta
            mcoords[i] -= delta
            pfunc = func(pcoords)
            mfunc = func(mcoords)
            L.append((pfunc-mfunc)/(2*delta))
        return np.array(L)

    @staticmethod
    def hess_2pt(func, coords, delta=0.0001):
        """
        func returns an array-like object
        """
        # TODO: the two calls to grad_2pt result in redundant calcs
        L = []
        num_coords = len(coords)
        for i in range(num_coords):
            pcoords = coords.copy()
            mcoords = coords.copy()
            pcoords[i] += delta
            mcoords[i] -= delta
            pgrad = numerical_grad.grad_2pt(func,pcoords,delta=delta)
            mgrad = numerical_grad.grad_2pt(func,mcoords,delta=delta)

            L.append((pgrad-mgrad)/(2*delta))

        return np.array(L)

    @staticmethod
    def hess_3pt(func, coords, delta=0.0001):
        """
        func returns an array-like object
        """
        # oops, this is wrong
        raise NotImplementedError
        L = []
        num_coords = len(coords)
        cfunc = func(coords)
        for i in range(num_coords):
            pcoords = np.zeros_like(coords)
            mcoords = np.zeros_like(coords)
            pcoords = coords
            mcoords = coords
            pcoords[i] += delta
            mcoords[i] -= delta
            pfunc = func(pcoords)
            mfunc = func(mcoords)

            pgrad = (pfunc-cfunc)/(delta)
            mgrad = (cfunc-mfunc)/(delta)
            L.append((pgrad-mgrad)/(delta))

        return np.array(L)