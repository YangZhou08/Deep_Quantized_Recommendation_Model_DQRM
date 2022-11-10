#PBS -l select=2:ncpus=160 -lplace=excl 
conda init bash 
conda activate aikit-pt 
export LD_LIBRARY_PATH=~/anaconda3/lib 
mpirun -n 2 -l python /homes/yangzhou08/Training_DLRM_fast/example_multiple_cpu_dp_two.py 
