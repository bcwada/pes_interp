#!/bin/env python3
import argparse
import numpy as np
import os
from pathlib import Path
from math import pi

from sheppard_pes import Pes
from dynamics import trajectory
import lib.xyz as xyz

def parse():
    parser = argparse.ArgumentParser(
        description="Run a scan along the C-C-C-C dihedral")
    parser.add_argument("pes_files", type=Path,
                        help="Path containings points for the PES")
    parser.add_argument("tc_files", type=Path,
                        help="Path for electronic structure calculations")
    parser.add_argument("init_conds", type=Path,
                        help="Path detailing initial positions and velocities")
    parser.add_argument("destination", type=Path,
                        help="Path for output files")
    return parser.parse_args()

def get_energy(geom,dir_name,tc_files):
    "Run a TC job to obtain energy of current GEOMetry.

    GEOM is an instance of xyz.Geometry
    DIR_NAME is a Path to run the dedicated TC job in
    TC_FILES is a Path containing the necessary files tc.in and sbatch.sh"
    # launch terachem job
    tc_input=tc_files/"tc.in"
    sbatch = tc_files/"sbatch.sh"
    tc_input=tc_input.absolute()
    sbatch=sbatch.absolute()

    with conman.minimal_context(dir_name,tc_input,sbatch_input) as man:
        print("launching job")
        print(os.getcwd())
        geom.write_file("geom.xyz")
        man.launch()
        man.wait_for_job()
        return tcReader.gradient.from_file("tc.out").energy



def main():
    args = parse()

    shep = Pes.pes_from_folder(args.pes_files)

    # (H4-6)(C0)(C1)(H7-8)(C2)(H9-10)(C3)(H11-13)

    destination=args.destination.absolute()

    geom = xyz.Geometry.from_file(args.init_conds/"pos.xyz")

    angle=0
    res=60
    step=2*pi/res

    with open("./scan.dat",'w') as file:
        for t in range(0,res+1):
            tc_energy=get_energy(geom,destination/f"{t:06d}",args.tc_files)
            file.write(f"{angle} {shep.eval_point_geom(geom)} {tc_energy}\n")
            geom.rot_bond([3,9,10,11,12,13],1,2,step)
            angle+=step



if __name__ == "__main__":
    main()
