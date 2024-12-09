# Conditional Prompt Learning for Vision-Language Models (Co-CoOp, CVPR'22)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2203.05557)

We provide the scripts in [scripts/cocoop](../scripts/cocoop) to reproduce Co-CoOp results (CVPR'22).

## Generalization From Base to New Classes

This corresponds to the experiments in main paper Table 1.

You will need both `scripts/cocoop/vanilla_base2new_train.sh` and `scripts/cocoop/vanilla_base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have one input arguments, i.e., `DATASET`.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
# dataset=imagenet
sh scripts/cocoop/base2new_train.sh imagenet
sh scripts/cocoop/base2new_test.sh imagenet
```

## Cross-Dataset Transfer

This corresponds to the experiments in the main paper Table 2.

The relevant scripts are `scripts/cocoop/vanilla_xd_train.sh` and `scripts/cocoop/vanilla_xd_test.sh` where the `DATASET` variable is set to the default, namely `imagenet`. To train the model, run

```bash
sh scripts/cocoop/vanilla_xd_train.sh
```

Then, you can evaluate the model on other datasets, e.g.,

```bash
# dataset=caltech101
sh scripts/cocoop/vanilla_xd_eval.sh caltech101
```

## Domain Generalization

This corresponds to the experiments in the supplementay material Table 8.

The steps are similar to those discussed in "Cross-Dataset Transfer" except you evaluate the model on the variants of ImageNet, i.e., `imagenetv2`, `imagenet_sketch`, `imagenet_a` and `imagenet_r`.

For example,
```bash
# dataset=caltech101
sh scripts/cocoop/vanilla_xd_eval.sh imagenetv2
```