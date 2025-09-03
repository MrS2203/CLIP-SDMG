# CLIP-SDMG
This repository contains the source code of CLIP-SDMG [CLIP-SDMG:CLIP knowledge distillation based on semantic decoupling and  mask generation].



## Install

```
pip install -r requirements-training.txt
```



## Dataset preparation

### Conceptual Captions 3M 

OpenCLIP reads a CSV file with two columns: a path to an image, and a text caption. 

Our downloaded CC3M training set contains 2.89M images, and our CC3M validation set contains 13K images.


The generated `cc3m_train.csv` is:
```
title   filepath
XXXXXX  train/X/X.jpg
...     ...
```

The generated `cc3m_val.csv` is:
```
title   filepath
XXXXXX  val/X/X.jpg
...     ...
```



## Supplementary Materials

Further supplementary experiments and theoretical details can be found at the following link.

[https://ars.els-cdn.com/content/image/1-s2.0-S0950705125012353-mmc1.pdf](https://ars.els-cdn.com/content/image/1-s2.0-S0950705125012353-mmc1.pdf)



## Acknowledgement

Our codebase is bulit over [open_clip](https://github.com/mlfoundations/open_clip), an open-source codebase to run CLIP models.

We would appreciate it if our paper and repo are helpful to you!
```
@article{zhang2025clip,
  title={CLIP-SDMG: CLIP Knowledge Distillation based on semantic decoupling and mask generation},
  author={Zhang, Qian and Si, Zhicheng and Shao, Mingwen and Liang, Hong},
  journal={Knowledge-Based Systems},
  pages={114194},
  year={2025},
  publisher={Elsevier}
}
```