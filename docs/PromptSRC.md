# Self-regulating Prompts: Foundational Model Adaptation without Forgetting (PromptSRC, ICCV'23)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.06948)

We provide the scripts in [scripts/promptsrc](../scripts/promptsrc) to reproduce PromptSRC results (ICCV'23).

## Generalization From Base to New Classes

This corresponds to the experiments in main paper Table 1.

You will need both `scripts/promptsrc/base2new_train.sh` and `scripts/promptsrc/base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
# dataset=imagenet
sh scripts/promptsrc/vanilla_base2new_train.sh imagenet
sh scripts/promptsrc/vanilla_base2new_test.sh imagenet

# dataset=caltech101
sh scripts/promptsrc/vanilla_base2new_train.sh caltech101
sh scripts/promptsrc/vanilla_base2new_test.sh caltech101
```
