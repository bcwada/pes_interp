import argparse
from IPython import embed

import sys
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
import tools.point_extractor as extract


class common_test_funcs:
    @classmethod
    def pseudo(cls, coords):
        path = Path("./test/test_files/BuH.test.xyz")
        g = xyz.Geometry.from_file(path)
        ref = sheppard.Pes_Point.calc_inv_dist(g.coords.reshape(-1))
        z = sheppard.Pes_Point.calc_inv_dist(coords)
        v = z - ref
        return 0.5 * np.dot(v, v)


class generate_test_files:

    BuH_torsion_num_points = 36
    md_nth_geom = 100
    output_folder = Path("./test/generated_files")
    torsion_folder = output_folder/"torsion_files"

    @classmethod
    def generate_folders(cls):
        cls.output_folder.mkdir(parents=True, exist_ok=True)
        cls.torsion_folder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def generate_BuH_pseudo(cls):
        """
        construct artificial hessian for testing purpose using artificial potential function
        """
        path = Path("./test/test_files/BuH.test.xyz")
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
        path = Path("./test/test_files/BuH_anti.xyz")
        g = xyz.Geometry.from_file(path)
        rot_ang = np.pi * 2 * (1 / cls.BuH_torsion_num_points)
        for i in range(cls.BuH_torsion_num_points):
            g.bond_rot(rot_ang, (1, 2), [3, 9, 10, 11, 12, 13])
            g.write_file(cls.torsion_folder/f"torsion_{i}.xyz")

    @classmethod
    def generate_tc_BuH_torsion(cls):
        """
        run Terachem on the BuH torsion angle scan
        """
        for i in range(cls.BuH_torsion_num_points):
            g = xyz.Geometry.from_file(cls.torsion_folder/f"torsion_{i}.xyz")
            with conman.minimal_context(Path(cls.torsion_folder/f"torsion_{i}"),"./test/tc_files/frequencies/tc.in","./test/tc_files/frequencies/sbatch.sh") as man:
                g.write_file("geom.xyz")
                man.launch()
                man.wait_for_job()

    @classmethod
    def extract_tc_BuH_torsion(cls):
        """
        produces the .pt and .ex files along the torsion scan 
        """
        for i in range(cls.BuH_torsion_num_points):
            with conman.enter_dir(Path(cls.torsion_folder/f"torsion_{i}")):
                extract.point_from_files(Path("tc.out"),Path("geom.xyz"),Path("scr.geom/Hessian.bin"),Path("extracted"))

    @classmethod
    def generate_torsion_plot(cls, path):
        """
        generates a plot of the torsion PES
        """
        x_ax = 2*np.pi*np.array(range(cls.BuH_torsion_num_points))/cls.BuH_torsion_num_points
        geom_files = [cls.torsion_folder/f"torsion_{i}/geom.xyz" for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        f, ax = plt.subplots(1,1)
        ax.set_xlabel("torsion angle (radians)")
        ax.set_ylabel("energy")

        print("step1")
        tc_data = [tc.gradient.from_file(cls.torsion_folder/f"torsion_{i}/tc.out") for i in range(cls.BuH_torsion_num_points)]
        y_tc = [i.energy for i in tc_data]
        ax.scatter(x_ax,y_tc,marker="x",color="r", label="tc energies")

        print("step2")
        test_pes = sheppard.Pes.pes_from_folder(path)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="b", label="sheppard with only .pt files")

        print("step3")
        test_pes = sheppard.Pes.pes_from_folder(path, include_ex=True)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="g", label="sheppard with .pt and .ex files")

        # generate a PES along the torsion scan using masked ground truth values
        print("step4")
        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%2 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="cyan", label="sheppard with half of ref data")

        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%4 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="lime", label="sheppard with quarter of ref data")

        ax.legend()
        f.savefig("./test/generated_files/test_fig.png")

    @classmethod
    def generate_tc_md_BuH(cls):
        """
        run Terachm on geometries from the md
        """
        md_path = Path("./test/generated_files/md")
        for p in md_path.glob("run*"):
            hess_dir = p/"hess_dir"
            hess_dir.mkdir(exist_ok=True)
            md_geoms = xyz.combinedGeoms.readFile(p/"scr/coors.xyz")
            for i in range(0,len(md_geoms.geometries), cls.md_nth_geom):
                g = md_geoms.geometries[i]
                if not (hess_dir/str(i)).exists():
                    with conman.minimal_context(hess_dir/str(i),"./test/tc_files/frequencies/tc.in","./test/tc_files/frequencies/sbatch.sh") as man:
                        g.write_file("geom.xyz")
                        man.launch()

    @classmethod
    def extract_tc_md_BuH(cls):
        md_path = Path("./test/generated_files/md")
        for p in md_path.glob("run*"):
            hess_dir = p/"hess_dir"
            md_geoms = xyz.combinedGeoms.readFile(p/"scr/coors.xyz")
            for i in range(0,len(md_geoms.geometries), cls.md_nth_geom):
                print(i)
                with conman.enter_dir(hess_dir/str(i)):
                    if not Path("extracted.pt").exists() or Path("extracted.ex").exists():
                        extract.point_from_files(Path("tc.out"),Path("geom.xyz"),Path("scr.geom/Hessian.bin"),Path("extracted"))

    @classmethod
    def generate_torsion_plot_2(cls):
        """
        generates a plot of the torsion PES comparing against additional md simulation
        """
        x_ax = 2*np.pi*np.array(range(cls.BuH_torsion_num_points))/cls.BuH_torsion_num_points
        geom_files = [cls.torsion_folder/f"torsion_{i}/geom.xyz" for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        f, ax = plt.subplots(1,1)
        ax.set_xlabel("torsion angle (radians)")
        ax.set_ylabel("energy")

        print("step1")
        tc_data = [tc.gradient.from_file(cls.torsion_folder/f"torsion_{i}/tc.out") for i in range(cls.BuH_torsion_num_points)]
        y_tc = [i.energy for i in tc_data]
        ax.scatter(x_ax,y_tc,marker="x",color="r", label="tc energies")

        # print("step2")
        # test_pes = sheppard.Pes.new_pes()
        # md_path = Path("./test/generated_files/md")
        # for p in md_path.glob("run_00"):
        #     hess_dir = p/"hess_dir"
        #     md_geoms = xyz.combinedGeoms.readFile(p/"scr/coors.xyz")
        #     for i in range(0,len(md_geoms.geometries), cls.md_nth_geom):
        #         print(i)
        #         if Path(hess_dir/str(i)/"extracted.pt").exists():
        #             test_pes.add_point(hess_dir/str(i)/"extracted.pt")
        #         elif Path(hess_dir/str(i)/"extracted.ex").exists():
        #             test_pes.add_point(hess_dir/str(i)/"extracted.ex")
        # y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        # ax.plot(x_ax, y_shep, color="r", label="sheppard with new extracted files")

        # print("step3")
        # test_pes = sheppard.Pes.new_pes()
        # md_path = Path("./test/generated_files/md")
        # for p in md_path.glob("run_00"):
        #     hess_dir = p/"hess_dir"
        #     md_geoms = xyz.combinedGeoms.readFile(p/"scr/coors.xyz")
        #     for i in range(0,len(md_geoms.geometries), cls.md_nth_geom):
        #         print(i)
        #         if Path(hess_dir/str(i)/"extracted.pt").exists():
        #             test_pes.add_point(hess_dir/str(i)/"extracted.pt")
        # y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        # ax.plot(x_ax, y_shep, color="pink", label="sheppard with new extracted .pt files")

        # generate a PES along the torsion scan using masked ground truth values
        print("step4")
        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%2 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="cyan", label="sheppard with half of ref data")

        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%4 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="lime", label="sheppard with quarter of ref data")

        print("step5")
        test_pes = sheppard.Pes.new_pes()
        extracted_file = list((cls.torsion_folder/f"torsion_{12}").glob("extracted*"))[0]
        test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="gray", linestyle="-", label="single point reference")

        test_pes = sheppard.Pes.new_pes()
        extracted_file = list((cls.torsion_folder/f"torsion_{18}").glob("extracted*"))[0]
        test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="gray", linestyle="dotted", label="single point reference")

        ax.legend()
        f.savefig(cls.output_folder/"test_fig_2.png")

    @classmethod
    def debug(cls):
        # generate a PES along the torsion scan using masked ground truth values
        x_ax = 2*np.pi*np.array(range(cls.BuH_torsion_num_points))/cls.BuH_torsion_num_points
        geom_files = [cls.torsion_folder/f"torsion_{i}/geom.xyz" for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%2 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        embed()

        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%4 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]

    @classmethod
    def generate_all(cls):
        generate_test_files.generate_folders()
        generate_test_files.generate_BuH_torsion()
        generate_test_files.generate_BuH_pseudo()
        generate_test_files.generate_tc_BuH_torsion()
        generate_test_files.extract_tc_BuH_torsion()
        # run md
        # generate_test_files.generate_tc_md_BuH()
        # generate_test_files.extract_tc_md_BuH()



class sheppard_test(unittest.TestCase):
    def point_test(self, test_path, pes, places=7):
        test_geom = xyz.Geometry.from_file(test_path)
        pseudo_e = common_test_funcs.pseudo(test_geom.coords.reshape(-1))
        # print(f"evaluated pseudo at test: {pseudo_e}")
        e = pes.eval_point(test_geom.coords.reshape(-1))
        # print(f"evaluated sheppard at test: {e}")
        self.assertAlmostEqual(pseudo_e, e, places=places)

    def test_energy_0(self):
        # print("Energy Test 0: get energy back at same point")
        pes = sheppard.Pes.new_pes()
        pes.add_point("./test/generated_files/BuH.pt", symmeterize=False)
        test_path = Path("./test/test_files/BuH.test.xyz")
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
        test_path = Path("./test/test_files/BuH.test.xyz")
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
        test_path = Path("./test/test_files/BuH.test.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        sym_z = self.sympy_grads.calc_inv_dist(one_d_geom)
        ext_z = self.exact_grads.calc_inv_dist(one_d_geom)
        self.assertTrue(np.allclose(sym_z, ext_z))

    def test_inv_jacobian_0(self):
        test_path = Path("./test/test_files/BuH.test.xyz")
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
        test_path = Path("./test/test_files/BuH.test.xyz")
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
    args.add_argument("-z", action="store_true", help="Brandon's helper arg")
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
        print(sys.argv)
        del sys.argv[1:]
        unittest.main()
    if args.make_plot is not None:
        generate_test_files.generate_torsion_plot(args.make_plot)
    if args.z:
        #generate_test_files.generate_tc_md_BuH()
        #generate_test_files.extract_tc_md_BuH()
        generate_test_files.generate_torsion_plot_2()
        #generate_test_files.debug()
        # generate_test_files.generate_BuH_pseudo()



if __name__ == "__main__":
    main()
