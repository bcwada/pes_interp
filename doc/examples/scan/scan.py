#!/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from math import pi

from sheppard_pes import Pes
from dynamics import trajectory
import lib.xyz as xyz

def parse():
    parser = argparse.ArgumentParser(
        description="First hacky version of combined ES and PES interpolation code")
    parser.add_argument("pes_files", type=Path,
                        help="Path containings points for the PES")
    parser.add_argument("init_conds", type=Path,
                        help="Path detailing initial positions and velocities")
    parser.add_argument("destination", type=Path,
                        help="Path for output files")
    return parser.parse_args()


def main():
    args = parse()

    shep = Pes.pes_from_folder(args.pes_files)

    # (H4-6)(C0)(C1)(H7-8)(C2)(H9-10)(C3)(H11-13)

    geom = xyz.Geometry.from_file(args.init_conds/"pos.xyz")

    angle=0
    step=2*pi/100

    with open("",'w') as file:
    for t in range(0,101):
        file.write(f"{angle} {shep.eval_point_geom(geom)}\n")
        pos.rot_bond([3,9,10,11,12,13],1,2,step)
        angle+=step


if __name__ == "__main__":
    main()
