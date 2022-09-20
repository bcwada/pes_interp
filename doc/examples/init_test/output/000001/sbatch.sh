#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -J "vv"
#SBATCH --qos=gpu_normal 
#SBATCH --gres=gpu:1
#SBATCH --mem=8Gb
#SBATCH --fin=*
#SBATCH --fout=*

#cd $SCRATCH

module load TeraChem/2022.06-intel-2017.8.262-CUDA-11.4.1
terachem tc.in > tc.out 2> tc.err
