from functools import cached_property
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import itertools

import PesInterp.global_vars as global_vars
import PesInterp.lib.xyz as xyz
from .grads import Sympy_Grad
from .grads import Exact_Grad
from .grads import numerical_grad
from .point_processor import point_generator

# TODO: properly symmeterize the transformation matrix
import copy

LEVEL_SHIFT = 1e-100

@dataclass
class Pes_Point:
    energy: float
    coords: np.array
    grads: np.array
    freqs: np.array
    transform_matrix: np.array

    grads_source = Sympy_Grad(global_vars.NUM_ATOMS)
    # grads_source = Sympy_Grad(global_vars.NUM_ATOMS)
    calc_inv_dist = grads_source.calc_inv_dist

    @classmethod
    def from_file(cls, file_loc):
        with open(file_loc) as f:
            lines = f.readlines()
            energy = float(lines[0])
            coords = np.array([float(i) for i in lines[1].split()]).reshape((global_vars.NUM_ATOMS, 3))
            grads = np.array([float(i) for i in lines[2].split()])
            freqs = np.array([float(i) for i in lines[3].split()])
            trans_mat = np.array([float(i) for i in lines[4].split()]).reshape((3*global_vars.NUM_ATOMS - 6, global_vars.NUM_ATOMS*(global_vars.NUM_ATOMS-1)//2))
        return cls(energy, coords, grads, freqs, trans_mat)

    @classmethod
    def permute_coords(cls, coords, perm):
        """
        perm is a permutation with the same shape as global_vars.PERMUTATION_GROUP
        """
        copied_coords = copy.deepcopy(coords)
        for orig_group, permed_group in zip(global_vars.PERMUTATION_GROUPS,perm):
            for i, j in zip(orig_group, permed_group):
                copied_coords[j] = coords[i]
        return copied_coords

    @classmethod
    def permute_inv_dist(cls, arr, perm):
        """
        performs the appropriate perumtation on the natoms*(natoms-1)/2 array
        Note: also works on appropriately sized matrices
        """
        natoms = global_vars.NUM_ATOMS
        assert(len(arr) == natoms*(natoms-1)//2)
        copied_arr = copy.deepcopy(arr)
        # other_arr = copy.deepcopy(copied_arr)
        num_inds = natoms*(natoms-1)//2

        def get_ind(i,j):
            less = min(i,j)
            more = max(i,j)
            ind = num_inds-((natoms-less-1)*(natoms-less)//2) + more - less - 1
            return ind

        mat = []
        for i in range(natoms):
            l = []
            for j in range(natoms):
                if i == j:
                    l.append(np.zeros_like(arr[0]))
                else:
                    l.append(arr[get_ind(i,j)])
            mat.append(l)
        mat = np.array(mat)

        mat_orig_copy = copy.deepcopy(mat)
        for orig_group, permed_group in zip(global_vars.PERMUTATION_GROUPS, perm):
            for i, j in zip (orig_group, permed_group):
                mat[j] = mat_orig_copy[i]
        mat_orig_copy = copy.deepcopy(mat)
        for orig_group, permed_group in zip(global_vars.PERMUTATION_GROUPS, perm):
            for i, j in zip (orig_group, permed_group):
                mat[:,j] = mat_orig_copy[:,i]

        for i in range(natoms):
            for j in range(i+1,natoms):
                copied_arr[get_ind(i,j)] = mat[i,j]
        return copied_arr

    def permute_self(self, perm):
        self.coords = Pes_Point.permute_coords(self.coords, perm)
        self.transform_matrix = Pes_Point.permute_inv_dist(self.transform_matrix.T, perm).T

    @property
    def inv_dist(self):
        # Should only be called after the coordinates are properly permuted
        return Pes_Point.calc_inv_dist(self.coords.reshape(-1))

    def taylor_approx_from_coords(self, other_coords):
        other_inv_dist = Pes_Point.calc_inv_dist(other_coords.reshape(-1))
        return self.taylor_approx(other_inv_dist)

    def taylor_approx(self, other_inv_dist):
        eta_new = self.transform_matrix @ other_inv_dist
        eta_old = self.transform_matrix @ self.inv_dist
        eta_diff = eta_new - eta_old
        #print(np.dot(np.power(eta_diff, 2), self.freqs))
        #print(eta_diff)
        #print(self.freqs)
        return self.energy + np.dot(eta_diff, self.grads) + 0.5*np.dot(np.power(eta_diff, 2), self.freqs)

@dataclass
class Pes:
    point_list: list

    def __init__(self):
        self.point_list = []

    @classmethod
    def new_pes(cls):
        return Pes()

    @classmethod
    def pes_from_folder(cls, path, include_ex=False, symmeterize=True):
        fold = Path(path)
        pes = cls.new_pes()
        for i in fold.glob("*.pt"):
            pes.add_point(i, symmeterize=symmeterize)
        if include_ex:
            for i in fold.glob("*.ex"):
                pes.add_point(i, symmeterize=symmeterize)
        return pes 

    def add_point(self, path, symmeterize=True):
        if not symmeterize:
            self.point_list.append(Pes_Point.from_file(Path(path)))
        else:
            perms = [itertools.permutations(i) for i in global_vars.PERMUTATION_GROUPS]
            combs = itertools.product(*perms)
            for c in combs:
                pt = Pes_Point.from_file(Path(path))
                pt.permute_self(c)
                self.point_list.append(pt)

    @classmethod
    def weight(self, z1, z2, p=global_vars.WEIGHTING_PARAM, power=global_vars.WEIGHTING_POWER):
        # return 1 / (LEVEL_SHIFT + np.power(np.sum(np.power(z1 - z2, p)), 1 / p))
        return 1 / (LEVEL_SHIFT + np.power(np.linalg.norm(z1-z2, ord=2),power))

    def eval_point_geom(self, geom: xyz.Geometry):
        """Return Energy for a given GEOMetry.

        GEOM must be an instance of xyz.Geometry.

        """
        return self.eval_point(geom.coords.reshape(-1))

    def eval_point(self, coords):
        """Return Energy for given COORDS.

        COORDS should be a 1D array of cartesian geometries.

        """
        inv_dist = Pes_Point.calc_inv_dist(coords)
        # weights = np.zeros(len(self.point_list))
        # energies = np.zeros(len(self.point_list))
        weights = []
        energies = []
        # wtime = 0
        # etime = 0
        for ind,i in enumerate(self.point_list):
            # t1 = time.time()
            weights.append(Pes.weight(i.inv_dist, inv_dist))
            # t2 = time.time()
            energies.append(i.taylor_approx(inv_dist))
            # t3 = time.time()
            # wtime += (t2-t1)
            # etime += (t3-t2)
        # print(f"wtime: {wtime}")
        # print(f"etime: {etime}")
        # weights = np.array(weights)
        # energies = np.array(energies)
        weights = weights/np.sum(weights)
        return np.dot(weights,energies)

    def get_weights(self, coords, normalize=False, p=global_vars.WEIGHTING_PARAM):
        inv_dist = Pes_Point.calc_inv_dist(coords)
        weights = []
        for ind,i in enumerate(self.point_list):
            weights.append(Pes.weight(i.inv_dist, inv_dist, p=p))
        if normalize:
            weights = weights/np.sum(weights)
        return weights

    def get_weight_statistics(self, coords, normalize=False, p=global_vars.WEIGHTING_PARAM):
        inv_dist = Pes_Point.calc_inv_dist(coords)
        weights = []
        for ind,i in enumerate(self.point_list):
            weights.append(Pes.weight(i.inv_dist, inv_dist, p=p))
        if normalize:
            weights = weights/np.sum(weights)
        weights = np.array(weights)
        avg = np.average(weights)
        num = len(weights)
        maxi = np.max(weights)
        num_max = np.count_nonzero(weights == maxi)
        max_inds = []
        for i,w in enumerate(weights):
            if w == maxi:
                max_inds.append(i)
        print(f"avg:{avg} num:{num} max{maxi} num_max{num_max} \nmax_inds:{max_inds}")


    def eval_gradient(self, coords):
        return numerical_grad.grad_2pt(self.eval_point, coords)

if __name__ == "__main__":
    pass