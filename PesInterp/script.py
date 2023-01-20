#!/bin/env python3
import argparse
import numpy as np
from pathlib import Path

from sheppard_pes import Pes
from dynamics import trajectory
import lib.xyz as xyz

def parse():
    parser = argparse.ArgumentParser(
        description="First hacky version of combined ES and PES interpolation code")
    parser.add_argument("pes_files", type=Path,
                        help="Path containings points for the PES")
    parser.add_argument("tc_files", type=Path,
                        help="Path for electronic structure calculations")
    parser.add_argument("init_conds", type=Path,
                        help="Path detailing initial positions and velocities")
    parser.add_argument("timestep", type=float,
                        help="timestep in a.u.")
    parser.add_argument("duration", type=float,
                        help="Total time to run the simulation")
    parser.add_argument("destination", type=Path,
                        help="Path for output files")
    return parser.parse_args()

def write_file(energy, gradient, dest):
    with open(dest,'w') as f:
        f.write(str(energy)+"\n\n")
        for i in gradient:
            f.write(str(i)+"\n")

def main():
    args = parse()

    ###TODO TODO
    masses = np.array(
    [
        12,
        12,
        12,
        12,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ])
    elements = np.array(
    [
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
    ]
    )
    pos = xyz.Geometry.from_file(args.init_conds/"pos.xyz").coords
    vel = xyz.Geometry.from_file(args.init_conds/"vel.xyz").coords
    tc = args.tc_files/"tc.in"
    sbatch = args.tc_files/"sbatch.sh"

    shep = Pes.pes_from_folder(args.pes_files)
    traj = trajectory.from_init_conds(pos,vel,masses,elements,tc,sbatch,args.destination)
    traj.set_restart(True)
    while traj.t < args.duration:
        traj.step(args.timestep)
        pes_e = shep.eval_point(traj.x.reshape(-1))
        pes_grad = shep.eval_gradient(traj.x.reshape(-1))
        write_file(pes_e,pes_grad,args.destination / f"pes{traj.num_calcs:06d}.eg")


if __name__ == "__main__":
    main()
