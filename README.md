# Fast Adversarial Training with Adaptive Step Size

This repository contains the code for reproducing ATAS, of our paper: [Fast Adversarial Training with Adaptive Step Size](https://arxiv.org/abs/2206.02417)


## Prerequisites
The code is tested under the following environment and it should be compatible with other versions of python and pytorch.
- python 3.6.8
- pytorch 1.4.0, torchvision 0.5.0
- Install autoattack with 
```pip install git+https://github.com/fra31/auto-attack```


## Instructions to Run the Code

Please first change the ROOT in `data.py` to your data folder. Then run 

```
bash run.sh
```

The results will be stored in `results/` with the saved models and their test accuracies. The log will be saved under `log/`

## Code Overview

The directory `models` contains model architecture definition files. The functions of the python files are:

- `ATAS.py`: code for training with ATAS.
- `attack.py`: code for the evaluation of our model. 
- `data.py`: data loading.
- `adv_attack.py`: generating adversarial examples for the training.
- `data_aug.py`: data augmentation and inverse data augmentation.

The functions of the scripts are
- `ATAS_CIFAR.sh`: running the training and evaluation for CIFAR10 and CIFAR100.
- `ATAS_ImageNet.sh`: running the training and evaluation for ImageNet.
- `run.sh`: example command for runing `ATAS_CIFAR.sh` and `ATAS_ImageNet.sh`

## Citation

```
@article{Huang2023ATAS,
    title={Fast Adversarial Training with Adaptive Step Size},
    author={Huang, Zhichao and Fan, Yanbo and Liu, Chen and Zhang, Weizhong and Zhang, Yong Zhang and Salzmann, Mathieu and Süsstrunk, Sabine and Wang, Jue},
    booktitle={IEEE Transcations on Image Processing},
    year={2023},
}
```
