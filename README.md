# Far Away in the Deep Space: Dense Nearest-Neighbor-Based Out-of-Distribution Detection ([UNCV](https://uncv2023.github.io/) @ ICCV 2023)
Official pytorch implementation of: https://arxiv.org/abs/2211.06660

<img src="https://user-images.githubusercontent.com/16369841/228007868-517a5348-ba6f-486b-b340-9fec9b588072.jpg" width="600" />

## Dependencies
The code relies on `mmsegmentation` for the construction of models and datasets. Tested with `mmsegmentation==0.25.0`, `mmcv==1.6.0`, on `pytorch=1.13.1`.

## Data
The data configuration files are in `nn_od_configs/datasets`. Set the environment variable `DATASETS_ROOT` to suit your system.

## Models
The model configuration files are in `nn_od_configs/models`.

### Checkpoints
Some model checkpoints trained on Cityscapes, for the given model configurations:
* Segmenter ViT-B linear head: [weights](https://lmb.informatik.uni-freiburg.de/resources/binaries/dense_ood_knns/segmenter_deit-b_linear_cityscapes.pth)
* SETR Naive ViT-B: [weights](https://drive.google.com/file/d/1kGzdSLCazsbgZe0Y1Lo6sNwv9s5V3CAp/view?usp=sharing) (from https://github.com/fudan-zvg/SETR)
* SETR Naive ViT-L: [weights](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_vit-large_8x1_768x768_80k_cityscapes/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth) (from https://github.com/open-mmlab/mmsegmentation)
* UPerNet ConvNeXt-S: [weights](https://lmb.informatik.uni-freiburg.de/resources/binaries/dense_ood_knns/upernet_convnext_small_cityscapes.pth)

## Usage
To test on a dataset with the default kNN hyperparameters:

`python test_ood.py <MODEL_CONFIG> <MODEL_CHECKPOINT> <TEST_DATA_CONFIG> `

e.g.:
```
python test_ood.py nn_ood_configs/models/segmenter_deit-b_linear_cityscapes.py checkpoints/segmenter_deit-b_linear_cityscapes.pth nn_ood_configs/datasets/roadanomaly.py
```

