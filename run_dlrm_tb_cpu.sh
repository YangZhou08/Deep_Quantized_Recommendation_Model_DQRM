#PBS -l select=1 lplace=excl 
export LD_LIBRARY_PATH=~/anaconda3/lib 
python dlrm_s_pytorch_pseudo_cpustb.py --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" --max-ind-range=10000000 --data-generation=dataset --data-set=terabyte --raw-data-file=/rscratch/data/terabyte_dataset/day --processed-data-file=/rscratch/data/terabyte_dataset/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=2048 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --test-freq=10240 --memory-map --quantization_flag --embedding_bit=4 --data-sub-sample-rate=0.875 >> ~/training1_log.txt 