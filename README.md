# Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving Data Stream
This is the implementation of ARCUS published in KDD 2022 [[paper](https://arxiv.org/abs/2206.04792)]


## Required packages
- Tensorflow 2.2.0
- Python 3.8.3
- Scikit-learn 0.23.1
- Numpy 1.18.5
- Pandas 1.0.5

## Data sets description and link
- The last column in each data set file refers to the anomaly label (1: anomaly, 0:normal)   </br>
- [Data sets link]([https://www.dropbox.com/sh/a3fhtp9zjjujrwa/AAD_4wkFaULuK-uJinbtw81Oa?dl=0](https://drive.google.com/drive/folders/1IeYCrG-eLJto25pfBQqZsXNeymQnLAvD?usp=sharing]) </br>
- The link includes the small data sets (also included in the repository) and large data sets exceeding 100MB </br>
<img src="figures/Data_sets.jpg" width="800"> 


## How to run ARCUS
### Parameters

- model_type: type of model, one of ["RAPP", "RSRAE", "DAGMM"]
- inf_type: type of inference, one of ["INC", "ADP"] where "INC" for incremental and "ADP" for adaptive (proposed)
- batch_size: batch size (default: 512)
- min_batch_size: min batch size (default: 32)
- init_epoch: initial number of epochs for creating models (default: 5)
- intm_epoch: interim number of epochs for training models after initialization  (default: 1)
- hidden_dim: latent dimensionality of AE (default: the number of pricipal component explaining at least 70% of variance)
- layer_num: the number of layers in AE

### Training script
```
$ python main.py --model_type RAPP --dataset_name MNIST_AbrRec --inf_type ADP --batch_size 512 --min_batch_size 32 --init_epoch 5 --intm_epoch 1 --hidden_dim 24 --layer_num 3 --learning_rate 1e-4 --reliability_thred 0.95 --similarity_thred 0.80 --seed 42 --gpu '0' 
----------------------------
Data set: MNIST_AbrRec
Model type:  RAPP
AUC: 0.909
```

## Example concept drift adaptation of ARCUS in INSECTS data sets
<img src="figures/Drift_adaptation.jpg" width="900">

## Default model layer size (learning rate) used for ARCUS
<img src="figures/ARCUS_params.jpg" width="350">

## 5. Citation
```
@inproceedings{yoon2022arcus,
  title={Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving Data Stream},
  author={Yoon, Susik, and Lee, Youngjun, and Lee, Jae-Gil and Lee, Byung Suk},
  booktitle={Proceedings of the 28th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={--},
  year={2022}
}
```
