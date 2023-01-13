export PATH=$PATH:/opt/pbs/default/bin
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh
export LD_LIBRARY_PATH=~/anaconda3/lib 
bash run_dist.sh -np 4 -ppn 2 -f $PBS_NODEFILE python -u /homes/yangzhou08/Training_DLRM_fast/dlrm_s_pytorch_tb_dp_one.py --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" --max-ind-range=10000000 --data-generation=dataset --data-set=terabyte --raw-data-file=/tier2/utexas/yzhou/day --processed-data-file=/tier2/utexas/yzhou/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=2048 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --test-freq=10240 --memory-map --nepochs=5 --data-sub-sample-rate=0.875 