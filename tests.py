import argparse

import numpy as np
import unittest
from pathlib import Path
import time

import matplotlib
import matplotlib.pyplot as plt

import sheppard_pes as sheppard
import grads
import lib.xyz as xyz
import lib.tc_reader as tc
import point_processor
import lib.context_manager as conman


class common_test_funcs:
    @classmethod
    def pseudo(cls, coords):
        path = Path("./test/test_files/BuH.xyz")
        g = xyz.Geometry.from_file(path)
        ref = sheppard.Pes_Point.calc_inv_dist(g.coords.reshape(-1))
        z = sheppard.Pes_Point.calc_inv_dist(coords)
        v = z - ref
        return 0.5 * np.dot(v, v)


class generate_test_files:

    BuH_torsion_num_points = 36
    output_folder = Path("./test/generated_files")

    @classmethod
    def generate_folders(cls):
        Path("./test/tc_files").mkdir(parents=True, exist_ok=True)
        Path("./test/generated_files").mkdir(parents=True, exist_ok=True)

    @classmethod
    def generate_BuH_pseudo(cls):
        """
        construct artificial hessian for testing purpose using artificial potential function
        """
        path = Path("./test/test_files/BuH.xyz")
        g = xyz.Geometry.from_file(path)
        H = grads.numerical_grad.hess_2pt(
            common_test_funcs.pseudo, g.coords.reshape(-1)
        )
        calc = point_processor.point_generator(g, 0, np.zeros(42), H)
        calc.write_point("./test/generated_files/BuH.pt")

    @classmethod
    def generate_BuH_torsion(cls):
        """
        generates geometries around a torsion scan in BuH
        """
        path = Path("./test/test_files/BuH.xyz")
        g = xyz.Geometry.from_file(path)
        rot_ang = np.pi * 2 * (1 / cls.BuH_torsion_num_points)
        for i in range(cls.BuH_torsion_num_points):
            g.bond_rot(rot_ang, (1, 2), [3, 9, 10, 11, 12, 13])
            g.write_file(f"./test/generated_files/torsion_{i}.xyz")

    @classmethod
    def generate_tc_BuH_torsion(cls):
        """
        run Terachem on the BuH torsion angle scan
        """
        for i in range(cls.BuH_torsion_num_points):
            g = xyz.Geometry.from_file(f"./test/generated_files/torsion_{i}.xyz")
            with conman.minimal_context(Path(f"./test/generated_files/torsion_{i}"),"./test/tc_files/tc.in","./test/tc_files/sbatch.sh") as man:
                g.write_file("geom.xyz")
                man.launch()
                man.wait_for_job()

    @classmethod
    def generate_torsion_plot(cls, path):
        """
        generates a plot of the torsion PES
        """
        x_ax = 2*np.pi*np.array(range(cls.BuH_torsion_num_points))/cls.BuH_torsion_num_points
        tc_data = [tc.gradient.from_file(Path(f"./test/generated_files/torsion_{i}/tc.out")) for i in range(cls.BuH_torsion_num_points)]
        y_tc = [i.energy for i in tc_data]
        f, ax = plt.subplots(1,1)
        #ax.scatter(x_ax,y_tc,marker="x",color="r")
        test_pes = sheppard.Pes.pes_from_folder(path)
        geom_files = [Path(f"./test/generated_files/torsion_{i}/geom.xyz") for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep)

        test_pes = sheppard.Pes.pes_from_folder(path, include_ex=True)
        geom_files = [Path(f"./test/generated_files/torsion_{i}/geom.xyz") for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="g")

        f.savefig("./test/generated_files/test_fig.png")


    @classmethod
    def generate_all(cls):
        generate_test_files.generate_folders()
        generate_test_files.generate_BuH_torsion()
        generate_test_files.generate_BuH_pseudo()
        generate_test_files.generate_tc_BuH_torsion()


class sheppard_test(unittest.TestCase):
    def point_test(self, test_path, pes, places=7):
        test_geom = xyz.Geometry.from_file(test_path)
        pseudo_e = common_test_funcs.pseudo(test_geom.coords.reshape(-1))
        # print(f"evaluated pseudo at test: {pseudo_e}")
        e = pes.eval_point(test_geom.coords.reshape(-1))
        # print(f"evaluated sheppard at test: {e}")
        self.assertAlmostEqual(pseudo_e, e, places=places)

    # def setUp(self):
    #     path = Path("./test/test_files/BuH.xyz")
    #     g = xyz.Geometry.from_file(path)
    #     pes = sheppard.Pes.new_pes()

    def test_test(self):
        self.assertEqual(1, 1)

    def test_energy_0(self):
        # print("Energy Test 0: get energy back at same point")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/generated_files/BuH.pt", symmeterize=False)
        test_path = Path("./test/test_files/BuH.xyz")
        self.point_test(test_path, pes)

    def test_energy_1(self):
        # print("Energy Test 0: single point")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/generated_files/BuH.pt", symmeterize=False)
        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        self.point_test(test_path, pes)

    def test_large_disp_0(self):
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/generated_files/BuH.pt", symmeterize=False)
        test_path = Path("./test/test_files/BuH.test.large_disp.xyz")
        self.point_test(test_path, pes)

    def test_energy_2(self):
        # print("Energy Test 2: single point")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/generated_files/BuH.pt")
        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        self.point_test(test_path, pes)

    def test_symmeterizer_0(self):
        # print("symmeterized at min")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/generated_files/BuH.pt", symmeterize=True)
        test_path = Path("./test/test_files/BuH.xyz")
        self.point_test(test_path, pes)

    def test_gradient_0(self):
        """Compare the numerical gradient from the pseudo potential function and the pes object"""
        # print("Gradient Test 1")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/generated_files/BuH.pt", symmeterize=False)
        test_path = Path("./test/test_files/BuH.test.large_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        # print(f"pseudo potential numerical grad: {grads.numerical_grad.grad_2pt(common_test_funcs.pseudo,test_geom.coords.reshape(-1))}")
        # print(f"pes numerical grad: {pes.eval_gradient(test_geom.coords.reshape(-1))}")
        pseudo_grad = grads.numerical_grad.grad_2pt(
            common_test_funcs.pseudo, test_geom.coords.reshape(-1)
        )
        pes_grad = pes.eval_gradient(test_geom.coords.reshape(-1))
        self.assertTrue(np.allclose(pseudo_grad, pes_grad, rtol=0, atol=10**-5))


class grads_test(unittest.TestCase):
    def setUp(self) -> None:
        num_atoms_BuH = 14
        self.sympy_grads = grads.Sympy_Grad(num_atoms_BuH)
        self.exact_grads = grads.Exact_Grad(num_atoms_BuH)

    def test_inv_dist_0(self):
        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_dist(one_d_geom)
        ext_z = self.exact_grads.calc_inv_dist(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))

    def test_inv_dist_1(self):
        test_path = Path("./test/test_files/BuH.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_dist(one_d_geom)
        ext_z = self.exact_grads.calc_inv_dist(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))

    def test_inv_jacobian_0(self):
        test_path = Path("./test/test_files/BuH.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_jacobian(one_d_geom)
        ext_z = self.exact_grads.calc_inv_jacobian(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))

    def test_inv_jacobian_1(self):
        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_jacobian(one_d_geom)
        ext_z = self.exact_grads.calc_inv_jacobian(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))


class hessian_test(unittest.TestCase):
    def setUp(self) -> None:
        num_atoms_BuH = 14
        self.sympy_grads = grads.Sympy_Grad(num_atoms_BuH)
        self.exact_grads = grads.Exact_Grad(num_atoms_BuH)

    def test_inv_hessian_0(self):
        test_path = Path("./test/test_files/BuH.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_hessian(one_d_geom)
        ext_z = self.exact_grads.calc_inv_hessian(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))

    def test_inv_hessian_1(self):
        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_hessian(one_d_geom)
        ext_z = self.exact_grads.calc_inv_hessian(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))


class timings:
    def time_inv_jacobian(self):
        num_atoms_BuH = 14
        self.exact_grads = grads.Exact_Grad(num_atoms_BuH)

        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        v1_start = time.time()
        ext_v1 = self.exact_grads.calc_inv_jacobian(one_d_geom)
        mid = time.time()
        ext_v2 = self.exact_grads.calc_inv_jacobian_alt(one_d_geom)
        v2_start = time.time()
        assert np.allclose(ext_v1, ext_v2)
        print(f"v1 time: {v1_start-mid}")
        print(f"v2 time: {mid-v2_start}")


def parse():
    args = argparse.ArgumentParser(description="run test cases for sheppard PES")
    args.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="generate the files necessary for testing",
    )
    args.add_argument(
        "--make_plot",
        type=Path,
        help="makes test plot of PES along BuH torsion scan using the pt files in the given path",
    )
    args.add_argument("-t", "--timings", action="store_true", help="run timing tests")
    args.add_argument("-u", "--unittest", action="store_true", help="perform unittests")
    args = args.parse_args()
    return args


def main():
    args = parse()
    if args.generate:
        generate_test_files.generate_all()
    if args.timings:
        timing = timings()
        timing.time_inv_jacobian()
    if args.unittest:
        # runs all unit tests
        # !!! Two of these tests fail at the moment, solely because of the level of precision that
        # the tests require are greater than what we should reasonbly expect, but I'm leaving the
        # tests as failing because they're a great reminder later that  we may just run into a precision issue
        unittest.main()
    if args.make_plot is not None:
        generate_test_files.generate_torsion_plot(args.make_plot)


if __name__ == "__main__":
    main()
