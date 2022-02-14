## Reliable Adversarial Distillation with Unreliable Teachers

Code for ICLR 2022 "[Reliable Adversarial Distillation with Unreliable Teachers](https://openreview.net/forum?id=u6TRGdzhfip&amp;referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions))" 

by *Jianing Zhu, Jiangchao Yao, Bo Han, Jingfeng Zhang, Tongliang Liu, Gang Niu, Jingren Zhou, Jianliang Xu, Hongxia Yang*.

Full code and instructions will be completed soon.

## Introduction

In this work, we found the soft-labels provided by the teacher model gradually becomes ***less and less reliable*** during the adversarial training of student model. Based on that,  we propose to ***partially trust*** the soft labels provided by the teacher model in adversarial distillation.

<img src="https://github.com/ZFancy/IAD/blob/main/pic/overview.png?raw=true" style="zoom:80%;" />

## Environment

- Python
- Pytorch
- CUDA
- Numpy

## Content

- ```./models```: models used for pre-train and distillation.
- ```./pre_train```: code for AT and ST.
- ```IAD-I.py```: Introspective Adversarial Distillation based on ARD.
- ```IAD-II.py```: Introspective Adversarial Distillation based on AKD2.

## Usage

**Pre-train**

- AT
```bash
cd ./pre_train
CUDA_VISIBLE_DEVICES='0' python AT.py --out-dir INSERT-YOUR-OUTPUT-PATH
```

- ST
```bash
cd ./pre_train
CUDA_VISIBLE_DEVICES='0' python ST.py --out-dir INSERT-YOUR-OUTPUT-PATH
```

**Distillation**

- ARD
```bash
CUDA_VISIBLE_DEVICES='0' python ARD.py --teacher_path INSERT-YOUR-TEACHER-PATH --out-dir INSERT-YOUR-OUTPUT-PATH
```

- AKD2
```bash
CUDA_VISIBLE_DEVICES='0' python AKD2.py --teacher_path INSERT-YOUR-TEACHER-PATH --out-dir INSERT-YOUR-OUTPUT-PATH
```

- IAD-I
```bash
CUDA_VISIBLE_DEVICES='0' python IAD-I.py --teacher_path INSERT-YOUR-TEACHER-PATH --out-dir INSERT-YOUR-OUTPUT-PATH
```

- IAD-II
```bash
CUDA_VISIBLE_DEVICES='0' python IAD-II.py --teacher_path INSERT-YOUR-TEACHER-PATH --out-dir INSERT-YOUR-OUTPUT-PATH
```

- basic eval
```bash
CUDA_VISIBLE_DEVICES='0' python basic_eval.py --model_path INSERT-YOUR-MODEL-PATH
```

## Citation

```bib
@inproceedings{zhu2022reliable,
title={Reliable Adversarial Distillation with Unreliable Teachers},
author={Jianing Zhu and Jiangchao Yao and Bo Han and Jingfeng Zhang and Tongliang Liu and Gang Niu and Jingren Zhou and Jianliang Xu and Hongxia Yang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=u6TRGdzhfip}
}
```

## Reference Code

[1] AT: https://github.com/locuslab/robust_overfitting

[2] TRADES: https://github.com/yaodongyu/TRADES/

[3] ARD: https://github.com/goldblum/AdversariallyRobustDistillation

[4] AKD2: https://github.com/VITA-Group/Alleviate-Robust-Overfitting

[5] GAIRAT: https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training

