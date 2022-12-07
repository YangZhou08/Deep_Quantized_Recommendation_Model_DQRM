#!/bin/bash
#PBS -l select=2:ncpus=112 -q clx 

function print_vars {
  for VAR in ${!CCL*} ${!I_MPI*} ${!i_mpi*} ${!KMP_*} ${!OMP_*} LD_PRELOAD ${!DLRM_*} ${!PYTORCH_*} ${!PCL_*} ${!LIBXSMM_*} ${!EMULATE_*} VIRTUAL_ENV ${!ARGS_*} $@ ; do
    if ! test -z ${!VAR} ; then
       echo "Using $VAR=${!VAR}"
    fi
  done
}

# train settings 
ARGS_NTASKS=8 # original -n or -np field 
ARGS_PPN=4 # original -ppn 
ARGS_HOSTFILE=$PBS_NODEFILE # original -f 

NNODES=`cat $ARGS_HOSTFILE | sort -u | wc -l` 
NP=$ARGS_NTASKS
OPT_HOSTFILE
