# Visual-Language Prompt Tuning with Knowledge-guided Context Optimization (KgCoOp, CVPR'23)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.13283)

We provide the scripts in [scripts/kgcoop](../scripts/cocoop) to reproduce KgCoOp results (CVPR'23).

## Generalization From Base to New Classes

This corresponds to the experiments in main paper Table 1.

You will need both `scripts/kgcoop/vanilla base2new_train.sh` and `scripts/kgcoop/vanilla_base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have one input arguments, i.e., `DATASET`

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
# dataset=imagenet
sh scripts/kgcoop/vanilla_base2new_train.sh imagenet
sh scripts/kgcoop/vanilla_base2new_test.sh imagenet

# dataset=caltech101
sh scripts/kgcoop/vanilla_base2new_train.sh caltech101
sh scripts/kgcoop/vanilla_base2new_test.sh caltech101
```

## Cross-Dataset Transfer

This corresponds to the experiments in the main paper Table 2.

The relevant scripts are `scripts/kgcoop/vanilla_xd_train.sh` and `scripts/kgcoop/vanilla_xd_test.sh` where the `DATASET` variable is set to the default, namely `imagenet`. To train the model, run

```bash
# dataset=caltech101
sh scripts/kgcoop/vanilla_xd_train.sh 1
```

Then, you can evaluate the model on other datasets, e.g.,

```bash
# dataset=caltech101
sh scripts/kgcoop/vanilla_xd_test.sh caltech101
```

## Domain Generalization

This corresponds to the experiments in the supplementary material Table 8.

The steps are similar to those discussed in "Cross-Dataset Transfer" except you evaluate the model on the variants of ImageNet, i.e., `imagenetv2`, `imagenet_sketch`, `imagenet_a` and `imagenet_r`.

For example,
```bash
# dataset=caltech101
sh scripts/kgcoop/vanilla_xd_test.sh imagenetv2
```
