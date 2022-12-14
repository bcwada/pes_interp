from functools import cached_property
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import os

import lib.xyz as xyz
import global_vars
from grads import Exact_Grad

@dataclass
class point_generator:
    geom: xyz.Geometry
    energy: float
    grad: np.array
    hess: np.array

    grad_source = Exact_Grad(global_vars.NUM_ATOMS)

    @cached_property
    def inv_dist(self):
        """
        returns the inverse interatomic bond distances

            Returns:
                nC2 numpy array where n is the number of atoms
        """
        return point_generator.grad_source.calc_inv_dist(self.geom.coords.reshape(-1))

    @cached_property
    def b(self):
        """
        returns the second derivative of z with respect to the coordinates

            Returns:
                nC2 x 3n numpy array where n is the number of atoms
        """
        return point_generator.grad_source.calc_inv_jacobian(self.geom.coords.reshape(-1))

    @cached_property
    def b2(self):
        """
        returns the second derivative of z with respect to the coordinates

            Returns:
                nC2 x 3n x 3n tensor (numpy array) where n is the number of atoms
        """
        return point_generator.grad_source.calc_inv_hessian(self.geom.coords.reshape(-1))

    @cached_property
    def u(self):
        """
        returns the transformation matrix from inverse interatomic distances to 3n-6 internal coordinates

            Returns:
                nC2 x 3n-6  tensor (numpy array) where n is the number of atoms
        """
        # TODO: verify that we don't need derivatives of U
        b = self.b
        u, s, v = np.linalg.svd(b)
        u_tilde = u[:, 0:3 * self.geom.num_atoms - 6]
        return u_tilde

    @cached_property
    def dzeta_dx(self):
        return np.tensordot(self.u, self.b, axes=([0],[0]))

    @cached_property
    def dzeta2_dx2(self):
        return np.tensordot(self.u, self.b2, axes=([0],[0]))

    @cached_property
    def dV_dzeta(self):
        """
        returns the potential with regards to the selected internal coords

            Returns:
                3n-6 array of the gradient
        """
        # TODO: Should solve the overconstrained optimization rather than truncate like this
        # a = self.dzeta_dx[:,:-6]
        # return np.linalg.solve(a.T, self.grad[:-6])
        return np.linalg.pinv(self.dzeta_dx.T) @ self.grad

    @cached_property
    def dV2_dzeta2(self):
        """
        returns the second derivative with regards to the selected internal coords

            Returns:
                3n-6 x 3n-6 array of the hessian
        """
        other_term = np.tensordot(self.dzeta2_dx2, self.dV_dzeta, axes=([0],[0]))
        mod_hess = self.hess - other_term
        dv2dzeta2 = np.linalg.pinv(self.dzeta_dx.T) @ mod_hess @ np.linalg.pinv(self.dzeta_dx)
        return dv2dzeta2

    @cached_property
    def _diag_hess(self):
        diag_dv2dz2, l = np.linalg.eigh(self.dV2_dzeta2)
        return diag_dv2dz2, l

    @cached_property
    def l(self):
        return self._diag_hess[1]

    @cached_property
    def frequencies(self):
        return self._diag_hess[0]

    @cached_property
    def m(self):
        m = self.l.T @ self.u.T
        return m

    @cached_property
    def dV_deta(self):
        return self.l.T @ self.dV_dzeta

    def write_point(self, filename):
        """Export internalized point to FILENAME.

        FILENAME should be the name of the file to be written, including the path.

        File extensions are ignored. The standard file extension is .pt (point file).
        Points with imaginary frequencies use .ex (exclude).

        """
        filename=os.path.splitext(filename)[0]
        if self.frequencies[0] < 0:
            filename=filename+".ex"
        else:
            filename=filename+".pt"
        with open(filename,'w') as f:
            #energey
            f.write(str(self.energy))
            f.write("\n")
            #coordinates
            for i in self.geom.coords.reshape(-1):
                f.write(str(i) + " ")
            f.write("\n")
            #grad in diagonalized Hessian frame
            for i in self.dV_deta:
                f.write(str(i) + " ")
            f.write("\n")
            #frequencies
            for i in self.frequencies:
                f.write(str(i) + " ")
            f.write("\n")
            #transformation matrix M from z to diagonal frame
            for i in self.m.reshape(-1):
                f.write(str(i) + " ")
            f.write("\n")



if __name__ == "__main__":
    print("testing point processor")
    path = Path("./test/point_processor/BuH.xyz")
    g = xyz.Geometry.from_file(path)
    calc = point_generator(g, 0, np.zeros(42), np.zeros((42,42)))

    calc.write_point("./test/point_processor/test.out")

#    print("shapes")
#    z = calc.z()
#    print(f"z {z.shape}")
#    b = calc.b()
#    print(f"b {b.shape}")
#    u = calc.u()
#    print(f"u {u.shape}")
#    b2 = calc.b2()
#    print(f"b2 {b2.shape}")
#    dzdx = calc.dzeta_dx()
#    print(f"dzdx {dzdx.shape}")
#    dz2dx2 = calc.dzeta2_dx2()
#    print(f"dz2dx2 {dz2dx2.shape}")
#    dvdzeta = calc.dV_dzeta()
#    print(f"dvdzeta {dvdzeta.shape}")
#    dv2dzeta2 = calc.dV2_dzeta2()
#    print(f"dv2dzeta2 {dv2dzeta2.shape}")

#    calc.get_pes_properties()
