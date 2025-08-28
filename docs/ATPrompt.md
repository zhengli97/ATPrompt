
# Instructions for reproducing our results.

In the \<Running\> section of readme.md, we provide detailed instructions for running CoOp+ATPrompt, and in the \<Training Logs & Weights\> section, we provide detailed training logs for your reference.

In the following, we will provide implementation details and hyperparameter settings for reproducing CoOp+ATPrompt, CoCoOp+ATPrompt, MaPLe+ATPrompt and DePT+ATPrompt. 

**💡 Important Note: Reproduction with the following settings may deviate or fluctuate from the reported values. This is due to the randomness of the training data partitioning (`oxford_pets.py line 77`). This is normal. We recommend that researchers run more experiments with different seeds to reproduce the corresponding results stably.**

(The experiments of DePT and PromptKD are not available yet. Since DePT and PromptKD are implemented differently from previous work, the migration will take some time.)

Below is the attribute table used for different datasets:

| Datasets | Attributes |
| :--: | :--: |
| ImageNet | color, shape |
| Caltech101 | shape, size |
| OxfordPets | playfulness, energy |
| Stanford Cars | luxury |
| Flowers102 | color, habitat, growth |
| Food101 | flavor, preparation |
| FGVC Aircraft | design, range |
| SUN 397 | function |
| DTD | pattern, color, design |
| EuroSAT | habitat |
| UCF101 | precision |

The above attributes correspond to the `cfg.TRAINER.ATPROMPT.ATT1_TEXT`, `cfg.TRAINER.ATPROMPT.ATT2_TEXT` and `cfg.TRAINER.ATPROMPT.ATT3_TEXT` variables in the code.

If you want to experiment with other attribute words, you can change the variable values ​​in the function defined in `train.py line 154`.

# Base-to-Novel Experiments

## CoOp+ATPrompt

In this experiment, keep other hyperparameters unchanged. For datasets including Caltech, OxfordPets, StanfordCars, Flowers, Food101, Aircraft, SUN397, EuroSAT and UCF101, we specifically set EPO=100, NCTX=2.

For the DTD dataset, set EPO=100, NCTX=4.

For the ImageNet dataset, set EPO=10, NCTX=2.

## CoCoOp+ATPrompt

In this experiment, keep other parameters unchanged. For datasets including ImageNet, Caltech, OxfordPets, Food101, FGVC Aircraft, SUN397 and DTD, we specifically set EPO=10, NCTX=2.

For the UCF-101 dataset, set EPO=10, NCTX=4.

For StanfordCars, Flowers and EuroSAT datasets, set EPO=10, NCTX=6.

## MaPLe+ATPrompt

In this experiment, keep other parameters unchanged. For datasets including Caltech101, OxfordPets, Flowers, EuroSAT, we specifically set EPO=10, NCTX=4.

For datasets including ImageNet, StanfordCars, Food101, SUN397, DTD, set EPO=5, NCTX=4.

For FGVC Aircraft and UCF101 datasets, set EPO=5, NCTX=2.

## DePT+ATPrompt

Note: There is a loss balance hyperparameter w in DePT, which is generally set to 0.7 by default.

In this experiment, other parameters are kept unchanged. For the datasets including Caltech, DTD, EuroSAT, FGVC Aircraft, Food101, Flowers, SUN397, and UCF101, we set EPO=10, NCTX=4, where w=0.6 is set for Caltech and DTD datasets, and w=0.5 is set for UCF101.

For StanfordCars and OxfordPets datasets, we set EPO=10, NCTX=2, where w=0.6 is set for StanfordCars.

# Cross-dataset & Domain Generalization Experiments

In this experimental setting, since the usual practice is to select a model trained on the source dataset and measure its generalization performance on other datasets. There will be performance fluctuations when reproducing it yourself, so we recommend that researchers run several seeds and select the best performing model for evaluation.

## CoOp+ATPrompt

In this experiment, other parameters are kept unchanged. On the ImageNet-1K source dataset, we set NCTX=4 and EPO=10.

## CoCoOp+ATPrompt

In this experiment, other parameters are kept unchanged. On the ImageNet-1K source dataset, we set NCTX=4 and EPO=5.

## MaPLe+ATPrompt

In this experiment, other parameters are kept unchanged. On the ImageNet-1K source dataset, we set NCTX=4 and EPO=2.

<hr/>

# 论文结果复现指引 (中文版)

在readme.md \<Running\>部分中，我们提供了详细的，用于运行CoOp+ATPrompt的执行指令，并在\<Training Logs & Weights\>部分，提供了详细的训练log用于参考。接下来，我们会提供用于复现CoOp+ATPrompt，CoCoOp+ATPrompt, MaPLe+ATPrompt的实现细节和超参数设定。(DePT和PromptKD的实验因为代码还没整合进来，暂时没有提供，DePT和PromptKD由于是用的不一样的实现框架，所以需要一定的时间进行迁移)。

**💡 重要提示：按照以下设定进行复现可能会与报告数值存在偏差或者波动，这是由于划分训练数据时的随机性引起的(`datasets/oxford_pets.py line 77`)，这是正常现象。我们推荐研究者多跑一些不同种子的实验以稳定复现对应的结果。**

以下是对于不同数据集所用的属性表：

| Datasets | Attributes |
| :--: | :--: |
| ImageNet | color, shape |
| Caltech101 | shape, size |
| OxfordPets | playfulness, energy |
| Stanford Cars | luxury |
| Flowers102 | color, habitat, growth |
| Food101 | flavor, preparation |
| FGVC Aircraft | design, range |
| SUN 397 | function |
| DTD | pattern, color, design |
| EuroSAT | habitat |
| UCF101 | precision |

这些属性分别对应代码中的`cfg.TRAINER.ATPROMPT.ATT1_TEXT`, `cfg.TRAINER.ATPROMPT.ATT2_TEXT`, `cfg.TRAINER.ATPROMPT.ATT3_TEXT`变量。

如果你想要尝试其他的属性词，可以更改`train.py line 154`所定义函数里的变量值。

# Base-to-Novel实验

## CoOp+ATPrompt

在该实验中，保持其他超参数不变，对于Caltech, OxfordPets, StanfordCars, Flowers, Food101, Aircraft, SUN397, EuroSAT, UCF101，我们特别地设定EPO=100, NCTX=2。

对于DTD, 设定EPO=100, NCTX=4。

对于ImageNet, 设定EPO=10, NCTX=2。

## CoCoOp+ATPrompt

在该实验中，保持其他参数不变，对于ImageNet, Caltech, OxfordPets, Food101, FGVC Aircraft, SUN397, DTD 设定EPO=10, NCTX=2。

对于UCF-101, 设定EPO=10, NCTX=4。

对于StanfordCars, Flowers, EuroSAT, 设定EPO=10, NCTX=6。

## MaPLe+ATPrompt

在该实验中，保持其他参数不变，对于Caltech101, OxfordPets, Flowers, EuroSAT, 设定EPO=10, NCTX=4。

对于ImageNet, StanfordCars, Food101, SUN397, DTD, 设定EPO=5, NCTX=4。

对于FGVC Aircraft, UCF101, 设定EPO=5, NCTX=2。

## DePT+ATPrompt

注：DePT中有一个损失平衡超参数w，一般默认设定为0.7。

在该实验中，保持其他参数不变，对于Caltech, DTD, EuroSAT, FGVC Aircraft, Food101, Flowers, SUN397, UCF101 设定EPO=10, NCTX=4, w=0.7 其中对于Caltech, DTD, 设定EPO=10, NCTX=4, w=0.6， 对于UCF101, 设定EPO=10, NCTX=4, w=0.5。

对于OxfordPets, 设定EPO=10, NCTX=2, w=0.7, 对于StanfordCars，设定EPO=10, NCTX=2, w=0.6。

# Cross-dataset & Domain Generalization 实验

在该实验设定下，因为是选择一个在源域上训练的模型在其他数据集上测泛化，在自己复现的时候会存在性能的波动，所以我们建议研究者多跑几个seed，选择其中较好的模型进行验证评估。

## CoOp+ATPrompt

在该实验中，保持其他参数不变，在ImageNet-1K源数据集上，设定NCTX=4, EPO=10。

## CoCoOp+ATPrompt

在该实验中，保持其他参数不变，在ImageNet-1K源数据集上，设定NCTX=4, EPO=5。

## MaPLe+ATPrompt

在该实验中，保持其他参数不变，在ImageNet-1K源数据集上，设定NCTX=4, EPO=2。
