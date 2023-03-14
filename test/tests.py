from IPython import embed
import cProfile
import pstats

import argparse
import itertools
import sys
import numpy as np
import unittest
from pathlib import Path
import time

import matplotlib
import matplotlib.pyplot as plt

import PesInterp.sheppard_pes as sheppard
import PesInterp.grads as grads
import PesInterp.lib.xyz as xyz
import PesInterp.lib.tc_reader as tc
import PesInterp.point_processor as point_processor
import PesInterp.lib.context_manager as conman
import PesInterp.tools.point_extractor as extract
import PesInterp.global_vars as global_vars

#region
#endregion

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
        (cls.output_folder/"sym_test").mkdir(parents=True, exist_ok=True)

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
                #man.wait_for_job()

    @classmethod
    def extract_tc_BuH_torsion(cls):
        """
        produces the .pt and .ex files along the torsion scan 
        """
        for i in range(cls.BuH_torsion_num_points):
            with conman.enter_dir(Path(cls.torsion_folder/f"torsion_{i}")):
                extract.point_from_files(Path("tc.out"),Path("geom.xyz"),Path("scr.geom/Hessian.bin"),Path("extracted"))

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
    def setup_torsion_plot(cls, plot_ref=True):
        """
        builds a plot and plots the terachem ground truth data
        """
        x_ax = 2*np.pi*np.array(range(cls.BuH_torsion_num_points))/cls.BuH_torsion_num_points
        geom_files = [cls.torsion_folder/f"torsion_{i}/geom.xyz" for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        f, ax = plt.subplots(1,1)
        ax.set_xlabel("torsion angle (radians)")
        ax.set_ylabel("energy")
        if plot_ref:
            tc_data = [tc.gradient.from_file(cls.torsion_folder/f"torsion_{i}/tc.out") for i in range(cls.BuH_torsion_num_points)]
            tc_data = np.array(tc_data)
            y_tc = [i.energy for i in tc_data]
            y_tc = np.array(y_tc) 
            ax.scatter(x_ax,y_tc,marker="x",color="r", label="tc energies")
            pt_inds = []
            ex_inds = []
            # TODO identify .pt and .ex points
        return x_ax, geoms, f, ax

    @classmethod
    def make_test_pes(cls, geoms, mod=1, ex=True, half=False, sym=False):
        """
        Evaluates points on a potential energy surface from part of 1D torsion scan data (generated with above functions)
        """
        test_pes = sheppard.Pes.new_pes()
        num_points = cls.BuH_torsion_num_points
        if half:
            num_points = num_points//2
        for i in range(num_points):
            if not i%mod == 0:
                continue
            if (cls.torsion_folder/f"torsion_{i}/extracted.pt").exists():
                test_pes.add_point(cls.torsion_folder/f"torsion_{i}/extracted.pt", symmeterize=sym)
            elif ex:
                if (cls.torsion_folder/f"torsion_{i}/extracted.ex").exists():
                    test_pes.add_point(cls.torsion_folder/f"torsion_{i}/extracted.ex", symmeterize=sym)
                else:
                    raise Exception("point data missing from torsion folder")
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        return y_shep
    
    @classmethod
    def generate_torsion_plot(cls, path, masks=False):
        """
        generates a plot of the torsion PES from a folder of points
        """
        x_ax = 2*np.pi*np.array(range(cls.BuH_torsion_num_points))/cls.BuH_torsion_num_points
        geom_files = [cls.torsion_folder/f"torsion_{i}/geom.xyz" for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        f, ax = plt.subplots(1,1)
        ax.set_xlabel("torsion angle (radians)")
        ax.set_ylabel("energy")

        print("step1: Plotting ground truth")
        tc_data = [tc.gradient.from_file(cls.torsion_folder/f"torsion_{i}/tc.out") for i in range(cls.BuH_torsion_num_points)]
        y_tc = [i.energy for i in tc_data]
        ax.scatter(x_ax,y_tc,marker="x",color="r", label="tc energies")

        print("step2: Plotting only .pt files")
        test_pes = sheppard.Pes.pes_from_folder(path)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="b", label="sheppard with only .pt files")

        print("step3: Plotting with both .pt and .ex files")
        test_pes = sheppard.Pes.pes_from_folder(path, include_ex=True)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="g", label="sheppard with .pt and .ex files")

        if masks:
            # generate a PES along the torsion scan using masked ground truth values
            print("step4: Plotting with masked ground truth (1/2 points)")
            test_pes = sheppard.Pes.new_pes()
            for i in range(cls.BuH_torsion_num_points):
                if not i%2 == 0:
                    continue
                extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
                test_pes.add_point(extracted_file)
            y_shep = [test_pes.eval_point_geom(g) for g in geoms]
            ax.plot(x_ax, y_shep, color="cyan", label="sheppard with half of ref data")

            print("step5: Plotting with masked ground truth (1/4 points)")
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
    def generate_torsion_plot_2(cls):
        """
        generates a plot of the torsion PES
        plots a single point, a symmeterized point, and every fourth point along the scan
        used to generate the test_fig_* files
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

        """
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
        """

        # generate a PES along the torsion scan using masked ground truth values
        print("step4")
        test_pes = sheppard.Pes.new_pes()
        print("adding points")
        for i in range(cls.BuH_torsion_num_points):
            if not i%4 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file)
        print("evaluating points")
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        print(len(y_shep))
        print("plotting")
        ax.plot(x_ax, y_shep, color="lime", label="sheppard with quarter of ref data")

        print("step5")
        test_pes = sheppard.Pes.new_pes()
        extracted_file = list((cls.torsion_folder/f"torsion_{12}").glob("extracted*"))[0]
        test_pes.add_point(extracted_file, symmeterize=False)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="gray", linestyle="-", label="single point reference, unsymmeterized")

        test_pes = sheppard.Pes.new_pes()
        extracted_file = list((cls.torsion_folder/f"torsion_{18}").glob("extracted*"))[0]
        test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="gray", linestyle="dotted", label="single point reference")

        ax.legend()
        f.savefig(cls.output_folder/"test_fig_2.png")

    @classmethod
    def generate_torsion_plot_3(cls, sym=True):
        """
        generates torsion scan plots testing inclusion of .pt and .ex files 
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

        # generate a PES along the torsion scan using masked ground truth values
        # print("step4")
        # test_pes = sheppard.Pes.new_pes()
        # for i in range(cls.BuH_torsion_num_points):
        #     if not i%1 == 0:
        #         continue
        #     if (cls.torsion_folder/f"torsion_{i}/extracted.pt").exists():
        #         test_pes.add_point(cls.torsion_folder/f"torsion_{i}/extracted.pt")
        # y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        # ax.plot(x_ax, y_shep, color="red", label="all .pt")

        print("step5")
        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%2 == 0:
                continue
            if (cls.torsion_folder/f"torsion_{i}/extracted.pt").exists():
                test_pes.add_point(cls.torsion_folder/f"torsion_{i}/extracted.pt", symmeterize=sym)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="green", label="1/2 .pt")

        print("test6")
        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%4 == 0:
                continue
            if (cls.torsion_folder/f"torsion_{i}/extracted.pt").exists():
                test_pes.add_point(cls.torsion_folder/f"torsion_{i}/extracted.pt", symmeterize=sym)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="blue", linestyle=":", label="1/4 .pt")

        print("test7")
        test_pes = sheppard.Pes.new_pes()
        for i in range(cls.BuH_torsion_num_points):
            if not i%4 == 0:
                continue
            extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
            test_pes.add_point(extracted_file, symmeterize=sym)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        ax.plot(x_ax, y_shep, color="red", linestyle=":", label="1/4 all")

        ax.legend()
        f.savefig(cls.output_folder/f"test_fig_3_p{global_vars.WEIGHTING_POWER}.png")

    @classmethod
    def generate_local_torsion_data(cls, sym=False, short=False, neighbors=False):
        """
        generates the local torsion plot

        creates a plot that verifies that neighboring points of a single point along the generated torsion scan are 
        qualitatively in good condition
        """
        x_ax, geoms, f, ax = cls.setup_torsion_plot()

        print("step1")
        tc_data = [tc.gradient.from_file(cls.torsion_folder/f"torsion_{i}/tc.out") for i in range(cls.BuH_torsion_num_points)]
        y_tc = [i.energy for i in tc_data]
        ax.scatter(x_ax,y_tc,marker="x",color="r", label="tc energies")

        num_points = cls.BuH_torsion_num_points
        if short:
            num_points = 7
        for i in range(num_points):
            print(i)
            test_pes = sheppard.Pes.new_pes()
            if neighbors:
                for j in range(max(0,i-1),min(i+2,cls.BuH_torsion_num_points)):
                    extracted_file = list((cls.torsion_folder/f"torsion_{j}").glob("extracted*"))[0]
                    test_pes.add_point(extracted_file, symmeterize=sym)
            else:
                extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
                test_pes.add_point(extracted_file, symmeterize=sym)
            y_shep = [test_pes.eval_point_geom(g) for g in geoms][max(i-2,0):i+3]
            ax.plot(x_ax[max(i-2,0):i+3], y_shep, linestyle="dotted", label=f"{i}")

        # ax.legend()
        f.savefig(cls.output_folder/f"loc_scan_sym={sym}_p={global_vars.WEIGHTING_POWER}_n={neighbors}")

    @classmethod
    def generate_torsion_sym_test(cls):
        """
        Generates another testing PES plot
        """
        x_ax, geoms, f, ax = cls.setup_torsion_plot()
        y_shep = cls.make_test_pes(geoms, mod=1, ex=True, half=True, sym=False)
        ax.plot(x_ax, y_shep, color="orange", label="mod1_exT_symF")
        y_shep = cls.make_test_pes(geoms, mod=1, ex=False, half=True, sym=False)
        ax.plot(x_ax, y_shep, color="blue", label="mod1_exF_symF")
        y_shep = cls.make_test_pes(geoms, mod=1, ex=True, half=True, sym=True)
        ax.plot(x_ax, y_shep, color="green", label="mod1_exT_symT")
        y_shep = cls.make_test_pes(geoms, mod=1, ex=False, half=True, sym=True)
        ax.plot(x_ax, y_shep, color="red", label="mod1_exF_symT")

        ax.legend()
        f.savefig(cls.output_folder/f"sym_test/p{global_vars.WEIGHTING_POWER}.png")

    @classmethod
    def debug(cls):
        """
        This is my debugging function and no one else's. Go write your own or delete this
        """
        def print_all(geom_list,pes):
            for i in range(len(geom_list)):
                print(i)
                pes.get_weight_statistics(geoms[i].coords.reshape(-1))
        geom_files = [cls.torsion_folder/f"torsion_{i}/geom.xyz" for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]

        pt_orig = sheppard.Pes_Point.from_file("./test/generated_files/BuH.pt")
        pt_edit = sheppard.Pes_Point.from_file("./test/generated_files/BuH.pt")
        perm_group = [[4,5,6],[8,7],[9,10],[11,12,13]]
        pt_edit.permute_self(perm_group)

        # test_pes = sheppard.Pes.new_pes()
        # for i in range(cls.BuH_torsion_num_points):
        #     if not i%4 == 0:
        #         continue
        #     extracted_file = list((cls.torsion_folder/f"torsion_{i}").glob("extracted*"))[0]
        #     test_pes.add_point(extracted_file)
        #y_shep = [test_pes.eval_point_geom(g) for g in geoms]

        test_pes = sheppard.Pes.new_pes()
        extracted_file = list((cls.torsion_folder/f"torsion_{18}").glob("extracted*"))[0]
        test_pes.add_point(extracted_file)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        # embed()

    @classmethod
    def generate_all_torsion(cls):
        # generate torsion data
        print("creating directories")
        generate_test_files.generate_folders()
        print("generating BuH geometries")
        generate_test_files.generate_BuH_torsion()
        print("generating pseudo-potential files")
        generate_test_files.generate_BuH_pseudo()
        print("submitting terachem jobs")
        generate_test_files.generate_tc_BuH_torsion()
    
    @classmethod
    def generate_all_md(cls):
        generate_test_files.generate_tc_md_BuH()

    @classmethod
    def make_plots(cls):
        #TODO
        pass

    @classmethod
    def generate_all(cls):
        cls.generate_all_torsion()
        cls.generate_all_md()

class profiler():

    BuH_torsion_num_points = 36
    md_nth_geom = 100
    output_folder = Path("./test/generated_files")
    torsion_folder = output_folder/"torsion_files"

    @classmethod
    def single_sym_point(cls):
        geom_files = [cls.torsion_folder/f"torsion_{i}/geom.xyz" for i in range(cls.BuH_torsion_num_points)]
        geoms = [xyz.Geometry.from_file(f) for f in geom_files]
        test_pes = sheppard.Pes.new_pes()
        extracted_file = list((cls.torsion_folder/f"torsion_{18}").glob("extracted*"))[0]
        test_pes.add_point(extracted_file, symmeterize=True)
        y_shep = [test_pes.eval_point_geom(g) for g in geoms]
        # y_shep = test_pes.eval_point_geom(geoms[10])

    @classmethod
    def profile(cls, func):
        profile = cProfile.Profile()
        profile.runcall(func)
        ps = pstats.Stats(profile)
        ps.print_stats()

class sheppard_test(unittest.TestCase):
    def point_test(self, test_path, pes, places=7):
        test_geom = xyz.Geometry.from_file(test_path)
        pseudo_e = common_test_funcs.pseudo(test_geom.coords.reshape(-1))
        # print(f"evaluated pseudo at test: {pseudo_e}")
        e = pes.eval_point(test_geom.coords.reshape(-1))
        # print(f"evaluated sheppard at test: {e}")

        # inv_dist = sheppard.Pes_Point.calc_inv_dist(test_geom.coords.reshape(-1))
        # weights = []
        # for i in pes.point_list:
        #     weights.append(pes.weight(i.inv_dist, inv_dist))
        # embed()

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

    def test_weight(self):
        pes = sheppard.Pes.new_pes()
        pt_orig = sheppard.Pes_Point.from_file("./test/generated_files/BuH.pt")
        pt_edit = sheppard.Pes_Point.from_file("./test/generated_files/BuH.pt")
        perm_group = [[4,5,6],[8,7],[9,10],[11,12,13]]
        pt_edit.permute_self(perm_group)
        self.assertEqual(pes.weight(pt_orig.inv_dist, pt_orig.inv_dist), 1e100)

class pes_point_test(unittest.TestCase):
    def test_permute(self) -> None:
        pt = sheppard.Pes_Point.from_file("./test/generated_files/BuH.pt")
        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        g = xyz.Geometry.from_file(test_path)
        perm_group = [[4,5,6],[8,7],[9,10],[11,12,13]]
        unperm_energy = pt.taylor_approx_from_coords(g.coords.reshape(-1))
        perm_g = sheppard.Pes_Point.permute_coords(g.coords, perm_group)
        pt.permute_self(perm_group)
        perm_energy = pt.taylor_approx_from_coords(perm_g)
        self.assertAlmostEqual(perm_energy, unperm_energy, delta=1e-7) # 0.02 cm^-1

    def test_permute_2(self) -> None:
        pt = sheppard.Pes_Point.from_file("./test/generated_files/BuH.pt")
        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        g = xyz.Geometry.from_file(test_path)
        perm_group = [[6,5,4],[8,7],[9,10],[11,12,13]]
        unperm_energy = pt.taylor_approx_from_coords(g.coords.reshape(-1))
        perm_g = sheppard.Pes_Point.permute_coords(g.coords, perm_group)
        pt.permute_self(perm_group)
        perm_energy = pt.taylor_approx_from_coords(perm_g)
        self.assertAlmostEqual(perm_energy, unperm_energy, delta=5e-7) # 0.1 cm^-1

    def test_permute_3(self) -> None:
        test_path = Path("./test/test_files/BuH.test.xyz")
        g = xyz.Geometry.from_file(test_path)
        perm_group = [[6,4,5],[8,7],[9,10],[11,12,13]]
        new_coords = sheppard.Pes_Point.permute_coords(g.coords, perm_group)
        self.assertTrue(np.all(new_coords[6] == g.coords[4]))

    def test_permute_4(self) -> None:
        l = []
        for i in range(global_vars.NUM_ATOMS):
            for j in range(i+1,global_vars.NUM_ATOMS):
                l.append((i,j))
        l = np.array(l)
        perm_group = [[6,5,4],[8,7],[9,10],[11,12,13]]
        reorder = sheppard.Pes_Point.permute_inv_dist(l, perm_group)

        # for i, j in zip(l,reorder):
        #     print(i,j)

    def test_symmetry(self) -> None:
        return
        p = Path("./test/generated_files/torsion_files/")
        g2 = sheppard.Pes_Point.from_file(p/f"torsion_2/extracted.pt")
        g3 = sheppard.Pes_Point.from_file(p/f"torsion_3/extracted.ex")
        g31 = sheppard.Pes_Point.from_file(p/f"torsion_31/extracted.ex")
        perm_group = [[4,6,5],[8,7],[10,9],[13,12,11]]
        g31.permute_self(perm_group)
        z2 = g2.inv_dist
        z3 = g3.inv_dist
        z31 = g31.inv_dist
        print("g2 and g3")
        print(z2-z3)
        print(sum(abs(z2-z3)))
        print("g3 and g31")
        print(z3-z31)
        print(sum(abs(z3-z31)))
        perms = [itertools.permutations(i) for i in global_vars.PERMUTATION_GROUPS]
        combs = itertools.product(*perms)
        mini = 10000
        for i in combs:
            g31 = sheppard.Pes_Point.from_file(p/f"torsion_31/extracted.ex")
            g31.permute_self(i)
            if sum(abs(g31.inv_dist - g3.inv_dist)) < mini:
                z31 = g31.inv_dist
                mini = sum(abs(g31.inv_dist - g3.inv_dist))
        print(mini)

        g2.permute_self(perm_group)
        print(sum(abs(g2.inv_dist - z2)))


# class grads_test(unittest.TestCase):
#     def setUp(self) -> None:
#         num_atoms_BuH = 14
#         self.sympy_grads = grads.Sympy_Grad(num_atoms_BuH)
#         self.exact_grads = grads.Exact_Grad(num_atoms_BuH)

#     def test_inv_dist_0(self):
#         test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
#         test_geom = xyz.Geometry.from_file(test_path)
#         one_d_geom = test_geom.coords.reshape(-1)
#         sym_z = self.sympy_grads.calc_inv_dist(one_d_geom)
#         ext_z = self.exact_grads.calc_inv_dist(one_d_geom)
#         self.assertTrue(np.allclose(sym_z, ext_z))

#     def test_inv_dist_1(self):
#         test_path = Path("./test/test_files/BuH.test.xyz")
#         test_geom = xyz.Geometry.from_file(test_path)
#         one_d_geom = test_geom.coords.reshape(-1)
#         sym_z = self.sympy_grads.calc_inv_dist(one_d_geom)
#         ext_z = self.exact_grads.calc_inv_dist(one_d_geom)
#         self.assertTrue(np.allclose(sym_z, ext_z))

#     def test_inv_jacobian_0(self):
#         test_path = Path("./test/test_files/BuH.test.xyz")
#         test_geom = xyz.Geometry.from_file(test_path)
#         one_d_geom = test_geom.coords.reshape(-1)
#         sym_z = self.sympy_grads.calc_inv_jacobian(one_d_geom)
#         ext_z = self.exact_grads.calc_inv_jacobian(one_d_geom)
#         self.assertTrue(np.allclose(sym_z, ext_z))

#     def test_inv_jacobian_1(self):
#         test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
#         test_geom = xyz.Geometry.from_file(test_path)
#         one_d_geom = test_geom.coords.reshape(-1)
#         sym_z = self.sympy_grads.calc_inv_jacobian(one_d_geom)
#         ext_z = self.exact_grads.calc_inv_jacobian(one_d_geom)
#         self.assertTrue(np.allclose(sym_z, ext_z))


# class hessian_test(unittest.TestCase):
#     def setUp(self) -> None:
#         num_atoms_BuH = 14
#         self.sympy_grads = grads.Sympy_Grad(num_atoms_BuH)
#         self.exact_grads = grads.Exact_Grad(num_atoms_BuH)

#     def test_inv_hessian_0(self):
#         test_path = Path("./test/test_files/BuH.test.xyz")
#         test_geom = xyz.Geometry.from_file(test_path)
#         one_d_geom = test_geom.coords.reshape(-1)
#         sym_z = self.sympy_grads.calc_inv_hessian(one_d_geom)
#         ext_z = self.exact_grads.calc_inv_hessian(one_d_geom)
#         self.assertTrue(np.allclose(sym_z, ext_z))

#     def test_inv_hessian_1(self):
#         test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
#         test_geom = xyz.Geometry.from_file(test_path)
#         one_d_geom = test_geom.coords.reshape(-1)
#         sym_z = self.sympy_grads.calc_inv_hessian(one_d_geom)
#         ext_z = self.exact_grads.calc_inv_hessian(one_d_geom)
#         self.assertTrue(np.allclose(sym_z, ext_z))


class timings:
    def time_inv_jacobian(self):
        num_atoms_BuH = 14
        self.exact_grads = grads.Exact_Grad(num_atoms_BuH)
        self.sym_grads = grads.Sympy_Grad(num_atoms_BuH)


        test_path = Path("./test/test_files/BuH.test.small_disp.xyz")
        test_geom = xyz.Geometry.from_file(test_path)
        one_d_geom = test_geom.coords.reshape(-1)
        t1 = time.time()
        ext_v1 = self.exact_grads.calc_inv_jacobian(one_d_geom)
        t2 = time.time()
        ext_v2 = self.exact_grads.calc_inv_jacobian_alt(one_d_geom)
        t3 = time.time()
        sym_v1 = self.sym_grads.calc_inv_jacobian(one_d_geom)
        t4 = time.time()
        sym_v1 = self.sym_grads.calc_inv_jacobian(one_d_geom)
        t5 = time.time()
        print(f"v1 time: {t2-t1}")
        print(f"v2 time: {t3-t2}")
        print(f"sym_1 time: {t4-t3}")
        print(f"sym_2 time: {t5-t4}")
        assert np.allclose(ext_v1, ext_v2)
        assert np.allclose(sym_v1, ext_v1)

class printouts:
    @classmethod
    def quantify_distances(cls, p_file_1, p_file_2):
        pt1 = sheppard.Pes_Point.from_file(p_file_1)
        pt2 = sheppard.Pes_Point.from_file(p_file_2)
        perms = [itertools.permutations(i) for i in global_vars.PERMUTATION_GROUPS]
        combs = itertools.product(*perms)
        combs = list(combs)

        max_norm_in_1 = -np.inf
        max_perm_in_1 = global_vars.PERMUTATION_GROUPS
        for i in combs:
            perm_point = sheppard.Pes_Point.from_file(p_file_1)
            perm_point.permute_self(i)
            if (dist := np.linalg.norm(pt1.inv_dist - perm_point.inv_dist)) > max_norm_in_1:
                max_norm_in_1 = dist
                max_perm_in_1 = i
        max_norm_in_2 = -np.inf
        max_perm_in_2 =  global_vars.PERMUTATION_GROUPS
        # for i in combs:
        #     perm_point = sheppard.Pes_Point.from_file(p_file_2)
        #     perm_point.permute_self(i)
        #     if (dist := np.linalg.norm(pt1.inv_dist - perm_point.inv_dist)) > max_norm_in_2:
        #         max_norm_in_2 = dist
        #         max_perm_in_2 = i
        max_norm_between = -np.inf
        min_norm_between = np.inf
        max_perm_between = None
        min_perm_between = None
        for i in combs:
            perm_point = sheppard.Pes_Point.from_file(p_file_2)
            perm_point.permute_self(i)
            if (dist := np.linalg.norm(pt1.inv_dist - perm_point.inv_dist)) > max_norm_between:
                max_norm_between = dist
                max_perm_between = i
            if (dist := np.linalg.norm(pt1.inv_dist - perm_point.inv_dist)) < min_norm_between:
                min_norm_between = dist
                min_perm_between = i

        # print("within group")
        # print(f"maximum norm: {max_norm_in_1}")
        # print(f"associated perm:{max_perm_in_1}")
        # print("between groups")
        # print(f"maximum norm: {max_norm_between}")
        # print(f"associated perm:{max_perm_between}")
        print(f"minimum norm: {min_norm_between}")
        print(f"associated perm:{min_perm_between}")

    @classmethod
    def quantify_torsion_distances(cls, ind1, ind2):
        f1 = list(Path(f"./test/generated_files/torsion_files/torsion_{ind1}").glob("extracted*"))[0]
        f2 = list(Path(f"./test/generated_files/torsion_files/torsion_{ind2}").glob("extracted*"))[0]
        cls.quantify_distances(f1, f2)

    @classmethod
    def quantify_assymetry(cls):
        # ref_geom = xyz.Geometry.from_file("./test/test_files/BuH_anti.xyz")
        # print(absref_geom.dist(0,5) - ref_geom.dist(0,6))
        pt = sheppard.Pes_Point.from_file(Path("./test/generated_files/torsion_files/torsion_0/extracted.pt"))
        pt2 = sheppard.Pes_Point.from_file(Path("./test/generated_files/torsion_files/torsion_35/extracted.pt"))
        # perm_group = [[4,6,5],[8,7],[10,9],[12,11,13]]
        # pt2.permute_self(perm_group)
        # print("sum abs diff of min geom and its mirror")
        # print(sum(abs(pt2.inv_dist - pt.inv_dist)))

        perms = [itertools.permutations(i) for i in global_vars.PERMUTATION_GROUPS]
        combs = itertools.product(*perms)
        z = None
        mini = 10000
        for i in combs:
            pt2 = sheppard.Pes_Point.from_file("./test/generated_files/torsion_files/"+f"torsion_35/extracted.pt")
            pt2.permute_self(i)
            if sum(abs(pt2.inv_dist - pt2.inv_dist)) < mini:
                z = pt2.inv_dist
                mini = sum(abs(pt2.inv_dist - pt.inv_dist))
        print(mini)

def parse():
    args = argparse.ArgumentParser(description="run test cases for sheppard PES")
    args.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="generate the files necessary for testing",
    )
    args.add_argument(
        "--generate_torsion",
        action="store_true",
        help="run terachem and generate point files along the BuH torsion scan"
    )
    args.add_argument(
        "--generate_md",
        action="store_true",
        help="run terachem and generate point files from BuH molecular dynamics"
    )
    args.add_argument(
        "--extract_torsion",
        action="store_true",
        help="extract data from terachem jobs for test torsion data",
    )
    args.add_argument(
        "--extract_md",
        action="store_true",
        help="extract data from terachem jobs for test md data",
    )
    args.add_argument(
        "--make_plots",
        action="store_true",
        help="from generated files generate relevant plots"
    )
    args.add_argument(
        "--make_plot",
        type=Path,
        help="makes test plot of PES along BuH torsion scan using the pt files in the given path",
    )
    args.add_argument("-p", "--profile", action="store_true", help="run some profiling on a PES")
    args.add_argument("-t", "--timings", action="store_true", help="run timing tests")
    args.add_argument("-u", "--unittest", action="store_true", help="perform unittests")
    args.add_argument("-z", action="store_true", help="developer debug argument")
    args = args.parse_args()
    return args


def main():
    args = parse()
    if args.generate:
        generate_test_files.generate_all()
    if args.generate_torsion:
        generate_test_files.generate_all_torsion()
    if args.generate_md:
        generate_test_files.generate_all_md()
    if args.extract_torsion:
        generate_test_files.extract_tc_BuH_torsion()
    if args.extract_md:
        generate_test_files.extract_tc_md_BuH()
    if args.make_plots:
        generate_test_files.make_plots()
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
        # generate_test_files.debug()
        # generate_test_files.generate_torsion_plot_2()
        generate_test_files.generate_torsion_plot_3()

    if args.profile:
        profiler.profile(profiler.single_sym_point)




if __name__ == "__main__":
    main()
