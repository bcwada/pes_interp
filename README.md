This is a library for building and interacting with Sheppard interpolated potential energy surfaces. Given molecular geometries, it can call Terachem through SLURM, diagonalize and symmeterize the Hessian in the local frame, and generate representations of each point suitable for a Sheppard interpolated PES. The PES can be used to evaluate the energy and gradient at new geometries and run basic MD.

Setup
-------------------
First, it is recommended to setup a new python environment. This library has been tested on python 3.9.7, so with that version you can run
~~~
python3 -m venv pes_env
~~~
and activate the environment with 
~~~
source pes_env/bin/activate
~~~
Once inside the environment, you can install the necessary versions with
~~~
pip install -r requirements.txt
~~~
all of these commands are in the setup.sh file, so alternately you can run 
~~~
source setup.sh
~~~

Alternately, if using Anaconda
~~~
conda create --name pes_env python=3.9.7
~~~
followed by
~~~
conda activate pes_env
pip install -r requirements.txt
~~~

Notes
-------------------
See Michael Colins and Keiran's paper
https://aip.scitation.org/doi/pdf/10.1063/1.476259