This is a library for building and interacting with Sheppard interpolated potential energy surfaces. Given molecular geometries, it can call Terachem through SLURM, diagonalize and symmeterize the Hessian in the local frame, and generate representations of each point suitable for a Sheppard interpolated PES. The PES can be used to evaluate the energy and gradient at new geometries and run basic MD.

Setup
-------------------
First, it is recommended to setup a new python environment
~~~
python -m venv pes_env
~~~
and use the environment with 
~~~
source activate pes_env/bin/activate
~~~
Once inside the environment, you can install the necessary versions with
~~~
pip install -r requirements.txt
~~~
all of these commands are in the setup.sh file, so alternately you can run 
~~~
source setup.sh
~~~

Notes
-------------------
Some equations used were based on Michael Colins and Keiran's paper
https://aip.scitation.org/doi/pdf/10.1063/1.476259