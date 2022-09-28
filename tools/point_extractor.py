#!/bin/env python3

import argparse
import os
from sys import argv
from pathlib import Path

import lib.xyz as xyz
import lib.tc_reader as tcReader
from point_processor import point_generator

# Prep parser
parser = argparse.ArgumentParser(description="Convert TeraChem output files to internal format.",usage=f"""{argv[0]} PES_FILES

Required files for each point are:
FILENAME.log: TeraChem output
FILENAME.xyz: Cartesian coordinate data
FILENAME.bin: Hessian binary file""")
parser.add_argument("pes_files", type=Path, help="folder of files to read in for the PES")

# get directory
args=parser.parse_args()
output_dir=args.pes_files


for log_file in output_dir.glob("*.log"):
    basename=str(log_file)
    basename=basename[:-4]
    xyz_file=Path(basename+".xyz")
    bin_file=Path(basename+".bin")
    if not os.path.exists(xyz_file):
        raise Exception("missing xyz file: {}".format(xyz_file))
    elif not os.path.exists(bin_file):
        raise Exception("missing bin file: {}".format(bin_file))
    q = xyz.Geometry.from_file(xyz_file)
    grad_obj=tcReader.gradient.from_file(log_file)
    E=grad_obj.energy
    grad=grad_obj.grad.reshape(-1)
    H=tcReader.Hessian.from_bin(bin_file).hess
    # #obtain E, grad, H from TC file see the parser in lib
    calc = point_generator(q, E, grad, H)
    calc.write_point(basename+".out")
