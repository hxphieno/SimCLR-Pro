# PyTorch SimCLR-Pro:SimCLR基于ViT的backbone改进
[![DOI](https://zenodo.org/badge/241184407.svg)](https://zenodo.org/badge/latestdoi/241184407)


### 

### 本项目旨在探讨结合卷积神经网络（CNN）与
Vision Transformer（ViT）架构在图像分类任务中的应用。我们提出了一种新的BackBone框架，该框架通过将ViT的局部特征提取能力与CNN的全局特征提取能力相结合，以期提升模型的性能。
此外，我们引入了 Squeeze-and-Excitation（SE）模块来进一步增强特征表示能力。通过在CIFAR10数据集上的实验，我们验证了所提出框架的有效性。
实验结果表明，与传统的ResNet架构以及单一的ViT架构相比，我们的模型在图像分类任务上展现出了更高的准确性和效率。本报告还包含了消融实验，以评估SE模块和CNN头部对模型性能的具体贡献。

## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./datasets --dataset-name cifar10 --log-every-n-steps 1 --epochs 100 

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.

For 16-bit precision GPU training, there **NO** need to to install [NVIDIA apex](https://github.com/NVIDIA/apex). Just use the ```--fp16_precision``` flag and this implementation will use [Pytorch built in AMP training](https://pytorch.org/docs/stable/notes/amp_examples.html).

## 
