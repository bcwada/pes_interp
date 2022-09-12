from functools import cached_property
import numpy as np
import sympy
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path

import tools.tcReader as tcReader
import tools.xyz as xyz
from tools.dynamics.grads import Sympy_Grad
from tools.dynamics.grads import Hard_Code_Grad

# based of Michael Colins and Keiran's papers
# https://aip.scitation.org/doi/pdf/10.1063/1.476259

@dataclass
class point_generator:
    geom: xyz.Geometry
    # TODO: make generat grad class to inherit from
    grad_source: Sympy_Grad

    def update_point(self, geom):
        self.geom = geom

    def z(self):
        """
        returns the inverse interatomic bond distances 

            Returns:
                nC2 numpy array where n is the number of atoms
        """
        return self.grad_source.calc_z()(*self.geom.coords.reshape(-1))

    def b(self):
        """
        returns the second derivative of z with respect to the coordinates

            Returns:
                nC2 x 3n numpy array where n is the number of atoms
        """
        return self.grad_source.calc_b()(self.geom.coords.reshape(-1))

    def b2(self):
        """
        returns the second derivative of z with respect to the coordinates

            Returns:
                nC2 x 3n x 3n tensor (numpy array) where n is the number of atoms
        """
        return self.grad_source.calc_b2()(self.geom.coords.reshape(-1))

    def u(self):
        """
        returns the transformation matrix from inverse interatomic distances to 3n-6 internal coordinates 

            Returns:
                nC2 x 3n-6  tensor (numpy array) where n is the number of atoms
        """
        # TODO: verify that we don't need derivatives of U
        b = self.b()
        u, s, v = np.linalg.svd(b)
        u_tilde = u[:, 0:3 * self.geom.num_atoms - 6]
        return u_tilde

    def dzeta_dx(self):
        return np.tensordot(self.u(), self.b(), axes=([0],[0]))

    def dzeta2_dx2(self):
        return np.tensordot(self.u(), self.b2(), axes=([0],[0]))

    def dV_dzeta(self, grad):
        """
        returns the potential with regards to the selected internal coords

            Returns:
                3n-6 array of the gradient
        """
        # TODO: Should solve the overconstrained optimization rather than truncate like this
        a = self.dzeta_dx()[:,:-6]
        return np.linalg.solve(a.T, grad[:-6])

    def dV2_dzeta2(self, grad, hessian):
        """
        returns the second derivative with regards to the selected internal coords

            Returns:
                3n-6 x 3n-6 array of the hessian
        """
        dvdzeta = self.dV_dzeta(grad)
        other_term = np.tensordot(self.dzeta2_dx2(), dvdzeta, axes=([0],[0]))
        mod_hess = hessian - other_term
        dzetadx = self.dzeta_dx()
        dv2dzeta2 = np.linalg.pinv(dzetadx.T) @ mod_hess @ np.linalg.pinv(dzetadx)
        return dv2dzeta2

    def get_pes_properties(self, grad, hessian):
        z = self.z()
        dvdz = self.dV_dzeta(grad)
        dv2dz2 = self.dV2_dzeta2(grad, hessian)
        diag_dv2dz2, l = np.linalg.eigh(dv2dz2)
        m = l.T @ self.u().T
        # z is the inverse atomic dist coords,
        # the second term should be dV/deta from eq. 4.6 of the paper
        # the diagonals of the hessian are stored, and the matrix to get from z to eta
        return z, l.T@dvdz, diag_dv2dz2, m


@dataclass
class Pes:
    # TODO: flush out setting and accessing parameters
    #debug: bool
    #geom_list: list
    z_list: list
    e_list: list
    grad_list: list
    hess_list: list
    m_list: list
    params: dict
    pnt_gen: point_generator
    #TODO: implement permutation invariance

    @classmethod
    def new(cls):
        return cls([],[],[],[],[],{"p":42}, point_generator(None,Sympy_Grad.initialize(g.num_atoms)))

    # @classmethod
    # def load_from_file_debug(cls, file):
    #     raise NotImplementedError

    @classmethod
    def load_from_file(cls, file):
        raise NotImplementedError

    # def save_debug(self, dest):
    #     raise NotImplementedError

    def save(self, dest):
        raise NotImplementedError

    def weight(self,z1,z2):
        np.power(np.power(z1-z2,self.params["p"]),1/self.params["p"])

    def taylor(self,z,pnt_ind):
        eta_new = self.m_list[pnt_ind] @ z
        eta_old = self.m_list[pnt_ind] @ self.z_list[pnt_ind]
        eta_diff = eta_new-eta_old
        return self.e_list[pnt_ind] + np.dot(eta_diff,self.grad_list[pnt_ind]) + np.dot(np.pow(eta_diff,2),self.hess_list[pnt_ind])

    def eval_point(self, geom):
        """
        given a new geometry, interpolates between all of the points already seen

            Returns:
                energy at the point
        """
        #TODO implement group symmetry
        self.pnt_gen.update_point(geom)
        z = self.pnt_gen.z()
        weights = []
        taylors = []
        for i in range(len(self.z_list)):
            weights.append(self.weight(z,self.z_list[i]))
            taylors.append(self.taylor(z,i))
        weight_sum = sum(weights)
        e = 0
        for i in range(len(self.z_list)):
            e += weights[i]*taylors[i]/weight_sum
        return e

    def add_point(self, calc, jobname="geom"):
        geom = xyz.Geometry.from_file(calc / f"{jobname}.xyz")
        #self.geom_list.append(geom)
        # little hack, the output file from the hessian calc first does a gradient at the point
        tc_grad = tcReader.gradient.from_file(calc / "tc.out")
        #self.grad_list.append(tc_grad.grad)
        tc_hess = tcReader.Hessian.from_bin(calc / f"scr.{jobname}/Hessian.bin")
        #self.hess_list.append(tc_hess.hess)
        self.pnt_gen.update_point(xyz.Geometry.from_file(calc/f"{jobname}.xyz"))
        z,g,h,m = self.pnt_gen.get_pes_properties(tc_grad.grad.reshape(-1),tc_hess.hess)
        self.z_list.append(z)
        self.e_list.append(tc_grad.energy)
        self.grad_list.append(g)
        self.hess_list.append(h)
        self.m_list.append(m)

if __name__ == "__main__":
    print("testing")
    path = Path("/home/bcwada/projects/grow/dynamics_sandbox/BuH.xyz")
    g = xyz.Geometry.from_file(path)
    calc = point_generator(g, Sympy_Grad.initialize(g.num_atoms))


    print("shapes")
    z = calc.z()
    print(f"z {z.shape}")
    b = calc.b()
    print(f"b {b.shape}")
    u = calc.u()
    print(f"u {u.shape}")
    b2 = calc.b2()
    print(f"b2 {b2.shape}")
    dzdx = calc.dzeta_dx()
    print(f"dzdx {dzdx.shape}")
    dz2dx2 = calc.dzeta2_dx2()
    print(f"dz2dx2 {dz2dx2.shape}")
    dvdzeta = calc.dV_dzeta(np.zeros(42))
    print(f"dvdzeta {dvdzeta.shape}")
    dv2dzeta2 = calc.dV2_dzeta2(np.zeros(42),np.zeros((42,42)))
    print(f"dv2dzeta2 {dv2dzeta2.shape}")

    calc.get_pes_properties(np.ones(42),np.ones((42,42)))