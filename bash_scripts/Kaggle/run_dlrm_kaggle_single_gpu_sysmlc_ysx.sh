# Important qat settings 
# quantization_flag means we use quantization-aware training here in the training script 
# quantize_act_and_lin means we quantize the linear layers but we don't quantize activation 
# embedding_bit the bitwidth used to quantize EMB with 
# weight_bit the bitwidth used to quantize mlp with 
# linear_channel channel-wise quantization is used for mlp quantization-aware training 
python /home/yzhou/Deep_Quantized_Recommendation_Model_DQRM/dlrm_s_pytorch_single_gpu_ysx.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --nepochs=5 --use-gpu --data-generation=dataset --data-set=kaggle --raw-data-file=/home/yzhou/dlrm_criteo_kaggle/train.txt --processed-data-file=/home/yzhou/dlrm_criteo_kaggle/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=1024 --save-model=/home/yzhou/dlrm_criteo_kaggle/save_model_after_training_one1.0.pt --quantization_flag --embedding_bit=4 --quant-mode=lsq --quantize_act_and_lin --weight_bit=4 --linear_channel --test-freq=10240 -n 1 -g 1 -nr 0 