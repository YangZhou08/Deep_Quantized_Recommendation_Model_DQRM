python -u /home/yzhou/Deep_Quantized_Recommendation_Model_DQRM/dlrm_s_pytorch_pseudo_multigpu.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --nepochs=5 --data-generation=dataset --data-set=kaggle --raw-data-file=/home/yzhou/dlrm_criteo_kaggle/train.txt --processed-data-file=/home/yzhou/dlrm_criteo_kaggle/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=512 --print-freq=1024 --print-time --test-mini-batch-size=1024 --use-gpu --save-model=/home/yzhou/dlrm_criteo_kaggle/save_model_after_training_one.pt  --test-freq=10240 --quantization_flag --quantize_act_and_lin --embedding_bit=4 --weight_bit=4 --linear_channel --number_of_gpus=4 