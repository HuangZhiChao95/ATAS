# The format for running ATAS_CIFAR.sh and ATAS_ImageNet.sh is
# bash ATAS_CIFAR.sh/ATAS_ImageNet.sh dataset epsilon

mkdir results
mkdir log
bash ATAS_CIFAR.sh cifar10 8
bash ATAS_CIFAR.sh cifar10 12
bash ATAS_CIFAR.sh cifar10 16
bash ATAS_CIFAR.sh cifar100 4
bash ATAS_CIFAR.sh cifar100 8
bash ATAS_CIFAR.sh cifar100 12
bash ATAS_ImageNet.sh imagenet 2
bash ATAS_ImageNet.sh imagenet 4
