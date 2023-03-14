#!/bin/env python3

import argparse
from os import path
from sys import argv
from pathlib import Path
import textwrap

import PesInterp.lib.xyz as xyz
import PesInterp.lib.tc_reader as tcReader
from PesInterp.point_processor import point_generator

def points_from_dir(tc_dir,output_dir=None):
    """Convert TeraChem output files in tc_dir to internal format.

    tc_dir is a path containing .log, .xyz and .bin files.
    output_dir is a path to write the .pt/.ex files (default tc_dir).

    Data points with imaginary frequencies are written to .ex files,
    otherwise they are written to .pt files.

    Both arguments are expected to be path-like objects.

    """
    if output_dir is None:
        output_dir=tc_dir
    for log_file in tc_dir.glob("*.log"):
        if __name__ == "__main__":
            print(f"Processing {log_file}...",end='')
        filename=path.splitext(log_file)[0]
        xyz_file=Path(filename+".xyz")
        bin_file=Path(filename+".bin")
        outname=output_dir/path.basename(filename)
        point_from_files(log_file,xyz_file,bin_file,outname)

def point_from_files(log_file,xyz_file,bin_file,outname):
    """Convert TeraChem output files to internal format.

    log_file is the TC output file.
    xyz_file is the cartesian point file.
    bin_file.xyz is the binary containing the Hessian.
    outname is the file to write to, sans extension.

    Data points with imaginary frequencies are written to .ex files,
    otherwise they are written to .pt files.

    All arguments are expected to be path-like objects.

    """
    if not path.exists(xyz_file):
        raise Exception("missing xyz file: {}".format(xyz_file))
    elif not path.exists(bin_file):
        raise Exception("missing bin file: {}".format(bin_file))
    q = xyz.Geometry.from_file(xyz_file)
    grad_obj=tcReader.gradient.from_file(log_file)
    E=grad_obj.energy
    grad=grad_obj.grad.reshape(-1)
    H=tcReader.Hessian.from_bin(bin_file).hess
    calc = point_generator(q, E, grad, H)
    calc.write_point(outname)
    if __name__ == "__main__" and (calc.frequencies[0] < 0):
        print(" Imaginary frequencies found!")
    elif __name__ == "__main__":
        print("")


if __name__ == "__main__":
    # Prep parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert TeraChem output files to internal format.",
        epilog=textwrap.dedent("""\
        Required files for each point are:
        FILENAME.log: TeraChem output
        FILENAME.xyz: Cartesian coordinate data
        FILENAME.bin: Hessian binary file

        Output files:
        FILENAME.pt: Valid data points
        FILENAME.ex: Excluded data points (imaginary frequencies)
        """))
    parser.add_argument("tc_output", type=Path, metavar="TC_OUTPUT",
                        help="Folder of files to read in for the PES")
    parser.add_argument("output_dir", type=Path, metavar="OUTPUT_DIR",
                        nargs='?',default=None,
                        help="Folder of files to write points to (default TC_OUTPUT)")
    # get directory
    args=parser.parse_args()
    print(f"Input dir: {args.tc_output}")
    print(f"Output dir: {args.output_dir or args.tc_output}")
    points_from_dir(args.tc_output,args.output_dir)
