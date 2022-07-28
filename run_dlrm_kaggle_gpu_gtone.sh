python dlrm_s_pytorch_dp_only.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --nepochs=5 --data-generation=dataset --data-set=kaggle --raw-data-file=/rscratch/data/dlrm_criteo/train.txt --processed-data-file=/rscratch/data/dlrm_criteo/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=128 --use-gpu --save-model=/rscratch/data/dlrm_criteo/save_model_after_training_one.pt --test-freq=10240 -n 1 -g 4 -nr 0 
