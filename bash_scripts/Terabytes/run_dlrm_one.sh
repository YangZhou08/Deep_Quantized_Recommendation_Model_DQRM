export PATH=$PATH:/opt/pbs/default/bin
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh
export LD_LIBRARY_PATH=~/anaconda3/lib 
bash run_dist.sh -np 2 -ppn 1 -f $PBS_NODEFILE hostname 