export PYTHONPATH="$PYTHONPATH:/Projects/pes_interp/pes_interp"
mkdir output

# arguments input PES file dir
# input TC dir
# initial conditions dir
# # timesteps
# output TC calculations
python /Projects/pes_interp/pes_interp/script.py pes_files/ tc_files/ init_conds/ 1 1 output
