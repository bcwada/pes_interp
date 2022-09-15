from functools import cached_property
from pathlib import Path
import numpy as np
from dataclasses import dataclass

import global_vars as global_vars
import lib.xyz as xyz
from grads import Sympy_Grad
from grads import numerical_grad
from point_processor import point_generator

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

    @cached_property
    def inv_dist(self):
        return Pes_Point.calc_inv_dist(self.coords.reshape(-1))

    def taylor_approx(self, other_inv_dist):
        eta_new = self.transform_matrix @ other_inv_dist
        eta_old = self.transform_matrix @ self.inv_dist
        eta_diff = eta_new - eta_old
        print(np.dot(np.power(eta_diff, 2), self.freqs))
        print(eta_diff)
        print(self.freqs)
        return self.energy + np.dot(eta_diff, self.grads) + np.dot(np.power(eta_diff, 2), self.freqs)


@dataclass
class Pes:
    point_list: list

    #TODO discuss, but I think permutation invariance is best implemented by 
    #making copies of each point with the permutation applied

    @classmethod
    def new_pes(cls):
        return Pes([])

    @classmethod
    def pes_from_folder(cls, path):
        fold = Path(path)
        pt_list = []
        for i in fold.glob("*"):
            pt_list.append(Pes_Point.from_file(i))

    def add_point(self, path, symmeterize=True):
        self.point_list.append(Pes_Point.from_file(Path(path)))

    def weight(self, z1, z2):
        return np.power(np.sum(np.power(z1 - z2, global_vars.WEIGHTING_PARAM)), 1 / global_vars.WEIGHTING_PARAM)

    def eval_point(self, geom: xyz.Geometry):
        # TODO symmeterize, grab gradients while we're at it
        inv_dist = Pes_Point.calc_inv_dist(geom.coords.reshape(-1))
        weights = []
        energies = []
        for i in self.point_list:
            weights.append(self.weight(i.inv_dist, inv_dist))
            energies.append(i.taylor_approx(inv_dist))
        weights = np.array(weights)
        energies = np.array(energies)
        weights = weights/np.linalg.norm(weights)

        return np.dot(weights,energies)

if __name__ == "__main__":
    print("testing pes")

    # construct artificial hessian for testing purpose
    # artificial hessian puts a harmonic potential on all interatomic distances
    path = Path("./test/sheppard_pes/BuH.xyz")
    g = xyz.Geometry.from_file(path)

    def pseudo(coords):
        ref = Pes_Point.calc_inv_dist(g.coords.reshape(-1))
        z = Pes_Point.calc_inv_dist(coords)
        v = z - ref
        return 0.5 * np.dot(v,v)

    H = numerical_grad.hess_3pt(pseudo, g.coords.reshape(-1))

    calc = point_generator(g, 0, np.zeros(42), H)
    calc.write_point("./test/sheppard_pes/BuH.out")    

    pes = Pes.new_pes()
    pes.add_point("./test/sheppard_pes/BuH.out")
    test_path = Path("./test/sheppard_pes/BuH.test.1.xyz")
    test_geom = xyz.Geometry.from_file(test_path)
    print(f"evaluated pseudo at test: {pseudo(test_geom.coords.reshape(-1))}")
    e = pes.eval_point(test_geom)
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