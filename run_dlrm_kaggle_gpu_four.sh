python dlrm_s_pytorch_comm_grad.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --nepochs=5 --data-generation=dataset --data-set=kaggle --raw-data-file=/home/yz25672/dlrm_criteo_kaggle/train.txt --processed-data-file=/home/yz25672/dlrm_criteo_kaggle/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=1024 --use-gpu --save-model=/home/yz25672/dlrm_criteo_kaggle/save_model_after_training_one.pt --quantization_flag --embedding_bit=4 --quantize_act_and_lin --weight_bit=4 --linear_channel --test-freq=10240 --quantize_embedding_bag_gradient --embedding_bag_gradient_bit_num=8 -n 1 -g 4 -nr 0 | tee four_gpus_grad8bit.txt 