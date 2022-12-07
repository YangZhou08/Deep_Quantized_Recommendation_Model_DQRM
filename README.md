DQRM: Deep Quantized Recommendation Model
=================================================================================
*A recommendation model that is small, powerful and efficient to train* 

## ![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png)`Please check out our project preprint`
[DQRM paper](./DQRM__Deep_learning_Quantized_Recommender_System_Model.pdf) 

![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png)`The project is submitting to KDD 2023`. 

Acknowledgement: 
------------
This repo is built based on the original [DLRM](https://github.com/facebookresearch/dlrm) repo, while the quantization code for embedding tables and linear layers is shared with [HAWQ](https://github.com/Zhen-Dong/HAWQ). 

Description: 
------------
State-of-the-art click through rate recommendation model DLRM is intensively used in the industry environment. However, DLRM has gigantic embedding tables which cause great memory overhead during inference and training. Large embedding tables took up more than 99% of the total parameters in these models, and previous works have shown that these embedding tables are over-parameterized. 

This project shows that large and over-parameterized embedding tables lead to severe overfitting and huge inefficiency in using the training data, as model overfits during second epoch of the training set. Then, the project shows that heavy quantization (uniform INT4 quantization) and quantization-aware training (QAT) can significantly reduce overfitting during training. The quantized model performance even edge against the original unquantized model on Criteo Kaggle Dataset. 

However, naive QAT can lead to inefficiency for recommendation models, exacebating the memory bottleneck during training. We proposed two techniques to improve the naive QAT. Also, DLRM are usually large and trained under distributed environments, we combined quantization and sparsification together to compress the communication. We publish our project as Deep Quantized Recommendation System (DQRM), which is a recommendation system that is small, powerful, and efficient to train. 

Results: 
------------
1) [Criteo Kaggle Dataset](https://ailab.criteo.com/ressources/) 

<img src="./kaggle_unquantized.png" width="900" height="320">
Heavily quantized models outperforms the original models by better overcoming overfitting (Criteo Kaggle Dataset)

| Settings    | Model bit width | Training loss | Testing Accuracy | Testing ROC AUC | 
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| baseline      |   FP32     | 0.303685 | 78.718% | 0.8001 | 
| DQRM   | INT4       | 0.436685 | 78.897% | 0.8035 | 

2) [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) 

<img src="./dlrm_terabyte.png" width="900" height="320">
Heavily quantized models outperforms the original models by better overcoming overfitting (Criteo Terabyte Dataset) 

<img src="./dlrm_terabyte_quantized_3.png" width="900" height="320">
Original model training loss falls as the model suffers overfitting, comparing with uniform INT4 Quantization-aware Training (QAT) 

| Settings    | Model bit width | Training loss | Testing Accuracy | Testing ROC AUC | 
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| baseline      |   FP32     | 0.347071 | 81.165% | 0.8004 | 
| DQRM   | INT4       | 0.412979 | 81.159% | 0.7998 | 

Running scripts 
------------
1) Running on GPUs and the Criteo Kaggle Dataset 
```
bash -x ./bash_scripts/Kaggle/run_dlrm_kaggle_gpu_four.sh 
``` 
2) Running on CPU clusters and the Criteo Terabyte Dataset (<span style="color:yellow">only simulation for multi-node available now, *real distributed environment will be online shortly*</span>) 
```
bash -x ./bash_scripts/Terabytes/run_dlrm_tb_cpu.sh 
``` 

Version
-------
0.1 : Initial release of the DLRM code

1.0 : DLRM with distributed training, cpu support for row-wise adagrad optimizer

Requirements
------------
pytorch-nightly (*11/10/20*)

scikit-learn

numpy

onnx (*optional*)

pydot (*optional*)

torchviz (*optional*)

mpi (*optional for distributed backend*)
