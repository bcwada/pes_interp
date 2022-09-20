from functools import cached_property
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import itertools

import global_vars
import lib.xyz as xyz
from grads import Sympy_Grad
from grads import numerical_grad
from point_processor import point_generator

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

    grads_source = Sympy_Grad.initialize(global_vars.NUM_ATOMS)
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

    @property
    def inv_dist(self):
        return Pes_Point.calc_inv_dist(self.coords.reshape(-1))

    def taylor_approx_from_coords(self, other_coords):
        # TODO properly symmeterize
        oc = copy.deepcopy(other_coords)
        self.perm(oc, self.perm_inds)
        print(oc)
        other_inv_dist = Pes_Point.calc_inv_dist(oc)
        return self.taylor_approx_from_inv_dist(other_inv_dist)

    def taylor_approx(self, other_inv_dist):
        eta_new = self.transform_matrix @ other_inv_dist
        eta_old = self.transform_matrix @ self.inv_dist
        eta_diff = eta_new - eta_old
        #print(np.dot(np.power(eta_diff, 2), self.freqs))
        #print(eta_diff)
        #print(self.freqs)
        return self.energy + np.dot(eta_diff, self.grads) + 0.5*np.dot(np.power(eta_diff, 2), self.freqs)

    def permute_coords(self,perm):
        """
        perm is a permutation with the same shape as global_vars.PERMUTATION_GROUP
        """
        # TODO: properly symmeterize
        self.perm_inds = perm
        return
        # maybe TODO, this is tightly coupled to global_vars now
        for orig_group, permed_group in zip(global_vars.PERMUTATION_GROUPS,perm):
            copied_coords = []
            for index in orig_group:
                copied_coords.append(self.coords[index].copy())
            for temp_index,index in enumerate(permed_group):
                self.coords[index] = copied_coords[temp_index]

    # TODO properly symmeterize
    def perm(self, other_list, perm):
        """
        perm is a permutation with the same shape as global_vars.PERMUTATION_GROUP
        """
        for orig_group, permed_group in zip(global_vars.PERMUTATION_GROUPS,perm):
            copied_coords = []
            for index in orig_group:
                copied_coords.append(other_list[index].copy())
            for temp_index,index in enumerate(permed_group):
                other_list[index] = copied_coords[temp_index]


@dataclass
class Pes:
    point_list: list

    @classmethod
    def new_pes(cls):
        return Pes([])

    @classmethod
    def pes_from_folder(cls, path):
        fold = Path(path)
        pt_list = []
        for i in fold.glob("*.out"):
            pt_list.append(Pes_Point.from_file(i))
        return cls(pt_list)

    def add_point(self, path, symmeterize=True):
        if not symmeterize:
            self.point_list.append(Pes_Point.from_file(Path(path)))
        else:
            perms = [itertools.permutations(i) for i in global_vars.PERMUTATION_GROUPS]
            combs = itertools.product(*perms)
            for c in combs:
                pt = Pes_Point.from_file(Path(path))
                pt.permute_coords(c)
                self.point_list.append(pt)

    def weight(self, z1, z2):
        return 1 / (LEVEL_SHIFT + np.power(np.sum(np.power(z1 - z2, global_vars.WEIGHTING_PARAM)), 1 / global_vars.WEIGHTING_PARAM))

    def eval_point_geom(self, geom: xyz.Geometry):
        inv_dist = Pes_Point.calc_inv_dist(geom.coords.reshape(-1))
        weights = []
        energies = []
        for i in self.point_list:
            weights.append(self.weight(i.inv_dist, inv_dist))
            energies.append(i.taylor_approx(inv_dist))
        weights = np.array(weights)
        energies = np.array(energies)
        weights = weights/np.sum(weights)
        return np.dot(weights,energies)

    def eval_point(self, coords):
        inv_dist = Pes_Point.calc_inv_dist(coords)
        weights = []
        energies = []
        for i in self.point_list:
            weights.append(self.weight(i.inv_dist, inv_dist))
            energies.append(i.taylor_approx(inv_dist))
        weights = np.array(weights)
        #print(weights)
        energies = np.array(energies)
        weights = weights/np.sum(weights)
        #print(weights)
        return np.dot(weights,energies)

    def eval_gradient(self, coords):
        return numerical_grad.grad_2pt(self.eval_point, coords)

if __name__ == "__main__":
    print("testing pes")

    path = Path("./test/sheppard_pes/BuH.xyz")
    g = xyz.Geometry.from_file(path)
    # artificial potential function puts a harmonic potential on all interatomic distances
    def pseudo(coords):
        ref = Pes_Point.calc_inv_dist(g.coords.reshape(-1))
        z = Pes_Point.calc_inv_dist(coords)
        v = z - ref
        return 0.5 * np.dot(v,v)

    def point_test(path, pes):
        name = path.absolute().name
        test_geom = xyz.Geometry.from_file(test_path)
        print("\n")
        print(f"running test with {name}")
        print(f"evaluated pseudo at test: {pseudo(test_geom.coords.reshape(-1))}")
        e = pes.eval_point(test_geom.coords.reshape(-1))
        print(f"evaluated sheppard at test: {e}")
        print("\n")

    # construct artificial hessian for testing purpose using artificial potential function

    # print("calculating numerical Hessian")
    # H = numerical_grad.hess_2pt(pseudo, g.coords.reshape(-1))
    # print("calculated numerical Hessian")

    # calc = point_generator(g, 0, np.zeros(42), H)
    # calc.write_point("./test/sheppard_pes/BuH.out")

    print("Energy Test 0: get energy back at same point")
    pes = Pes.new_pes()
    pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=False)
    test_path = Path("./test/sheppard_pes/BuH.xyz")
    point_test(test_path, pes)

    print("symmeterized at min")
    pes = Pes.new_pes()
    pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=True)
    point_test(test_path, pes)

    print("Energy Test 1: single point")
    pes = Pes.new_pes()
    pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=False)
    test_path = Path("./test/sheppard_pes/BuH.test.small_disp.xyz")
    point_test(test_path, pes)

    test_path = Path("./test/sheppard_pes/BuH.test.large_disp.xyz")
    point_test(test_path, pes)

    test_path = Path("./test/sheppard_pes/BuH.test.large_disp.xyz")
    test_geom = xyz.Geometry.from_file(test_path)
    print("\n")
    print("Gradient Test 1")
    print(f"pseudo potential numerical grad: {numerical_grad.grad_2pt(pseudo,test_geom.coords.reshape(-1))}")
    print(f"pes numerical grad: {pes.eval_gradient(test_geom.coords.reshape(-1))}")
    print("\n")
#
    print("Energy Test 2: single point")
    pes = Pes.new_pes()
    pes.add_point("./test/sheppard_pes/BuH.out")
    test_path = Path("./test/sheppard_pes/BuH.test.small_disp.xyz")
    test_geom = xyz.Geometry.from_file(test_path)
    print(f"evaluated pseudo at test: {pseudo(test_geom.coords.reshape(-1))}")
    e = pes.eval_point(test_geom.coords.reshape(-1))
    print(f"evaluated sheppard at test: {e}")



    # need to fix this for better test
    #global_vars.NUM_ATOMS = 4

    #path = Path("./test/sheppard_pes/c4.xyz")
    #g = xyz.Geometry.from_file(path)
    #calc = point_generator(g, 0, np.zeros(12), np.ones((12,12)))
    #calc.write_point("./test/sheppard_pes/c4.out")

    #pes = Pes.new_pes()
    #pes.add_point("./test/sheppard_pes/c4.out")

    #test_path = Path("./test/sheppard_pes/c4.test.2.xyz")
    #test_geom = xyz.Geometry.from_file(test_path)
    #e = pes.eval_point(test_geom)
    #print(e)
