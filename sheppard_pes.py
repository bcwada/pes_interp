"""
Just a scaffold at this point
"""

import numpy as np
from dataclasses import dataclass

import tools.tcReader as tcReader
import tools.xyz as xyz
from grads import Sympy_Grad


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