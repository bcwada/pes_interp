from IPython import embed

import numpy as np
import unittest
from pathlib import Path

import sheppard_pes as sheppard
import grads
import lib.xyz as xyz
import point_processor

class common_test_funcs():
    @classmethod
    def pseudo(cls, coords):
        path = Path("./test/sheppard_pes/BuH.xyz")
        g = xyz.Geometry.from_file(path)
        ref = sheppard.Pes_Point.calc_inv_dist(g.coords.reshape(-1))
        z = sheppard.Pes_Point.calc_inv_dist(coords)
        v = z - ref
        return 0.5 * np.dot(v,v)

class generate_test_files():
    @classmethod
    def generate_BuH_pseudo(cls):
        """ 
        construct artificial hessian for testing purpose using artificial potential function
        """
        path = Path("./test/sheppard_pes/BuH.xyz")
        g = xyz.Geometry.from_file(path)
        print("calculating numerical Hessian")
        H = grads.numerical_grad.hess_2pt(common_test_funcs.pseudo, g.coords.reshape(-1))
        print("calculated numerical Hessian")
        calc = point_processor.point_generator(g, 0, np.zeros(42), H)
        calc.write_point("./test/generated_files/BuH.out")

    @classmethod
    def generate_all(cls):
        generate_test_files.generate_BuH_pseudo()

class sheppard_test(unittest.TestCase):

    def point_test(self, test_path, pes, places=7):
        test_geom = xyz.Geometry.from_file(test_path)
        pseudo_e = common_test_funcs.pseudo(test_geom.coords.reshape(-1))
        # print(f"evaluated pseudo at test: {pseudo_e}")
        e = pes.eval_point(test_geom.coords.reshape(-1))
        # print(f"evaluated sheppard at test: {e}")
        self.assertAlmostEqual(pseudo_e, e, places=places)

    # def setUp(self):
    #     path = Path("./test/sheppard_pes/BuH.xyz")
    #     g = xyz.Geometry.from_file(path)
    #     pes = sheppard.Pes.new_pes()

    def test_test(self):
        self.assertEqual(1,1)

    def test_energy_0(self):
        # print("Energy Test 0: get energy back at same point")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=False)
        test_path = Path("./test/sheppard_pes/BuH.xyz")
        self.point_test(test_path, pes)

    def test_energy_1(self):
        # print("Energy Test 0: single point")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=False)
        test_path = Path("./test/sheppard_pes/BuH.test.small_disp.xyz")
        self.point_test(test_path, pes)
    
    def test_large_disp_0(self):
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=False)
        test_path = Path("./test/sheppard_pes/BuH.test.large_disp.xyz")
        self.point_test(test_path, pes)
    
    def test_energy_2(self):
        # print("Energy Test 2: single point")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/sheppard_pes/BuH.out")
        test_path = Path("./test/sheppard_pes/BuH.test.small_disp.xyz")
        self.point_test(test_path, pes)
        
    def test_symmeterizer_0(self):
        # print("symmeterized at min")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=True)
        test_path = Path("./test/sheppard_pes/BuH.xyz")
        self.point_test(test_path, pes)

    def test_gradient_0(self):
        """Compare the numerical gradient from the pseudo potential function and the pes object"""
        #print("Gradient Test 1")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/sheppard_pes/BuH.out", symmeterize=False)
        test_path = Path("./test/sheppard_pes/BuH.test.large_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        #print(f"pseudo potential numerical grad: {grads.numerical_grad.grad_2pt(common_test_funcs.pseudo,test_geom.coords.reshape(-1))}")
        #print(f"pes numerical grad: {pes.eval_gradient(test_geom.coords.reshape(-1))}")
        pseudo_grad = grads.numerical_grad.grad_2pt(common_test_funcs.pseudo,test_geom.coords.reshape(-1))
        pes_grad = pes.eval_gradient(test_geom.coords.reshape(-1))
        self.assertTrue(np.allclose(pseudo_grad, pes_grad,rtol=0,atol=10**-5))

class grads_test(unittest.TestCase):

    def setUp(self) -> None:
        num_atoms_BuH = 14
        self.sympy_grads = grads.Sympy_Grad(num_atoms_BuH)
        self.exact_grads = grads.Exact_Grad(num_atoms_BuH)

    def test_inv_dist_1(self):
        test_path = Path("./test/sheppard_pes/BuH.test.small_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_dist(one_d_geom)
        ext_z = self.exact_grads.calc_inv_dist(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))

    def test_inv_dist_2(self):
        test_path = Path("./test/sheppard_pes/BuH.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_dist(one_d_geom)
        ext_z = self.exact_grads.calc_inv_dist(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))

if __name__ == "__main__":
    # probably want to comment the following line out after running once
    # generate_test_files.generate_all()
    unittest.main()
