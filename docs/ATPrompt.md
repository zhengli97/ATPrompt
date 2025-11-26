# How to reproduce ATPrompt

## Preliminary

1. Create the environment and install Dassl.pytorch library. Please follow the instructions detailed in [[INSTALL.md](INSTALL.md)].

2. Prepare the dataset. Please follow the instructions detailed in [[DATASETS.md](DATASETS.md)]. For your download convenience, we have provided 14 datasets (excluding ImageNet-1K) in the Huggingface platform. [[HuggingFace_Download_Links](https://huggingface.co/zhengli97/prompt_learning_dataset)]

3. Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder. Comment the `trainers/coop.py line 42` and uncomment the `line 43`.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

## 🚀 Running ATPrompt

### Step I: Attribute Search (Optional)

**For more practical information about this process, please refer to [[Attribute_Search.md](Attribute_Search.md)].**

(1) Directly use our results.

Here we provide the five attribute bases obtained by querying the LLM (GPT-4o) and the final result after the differentiable attribute search. You can directly use our results for subsequent training.

Expand the list below👇 to see the results:
<details>
<summary>Click to expand "Attribute Lists"</Summary>

| Dataset | Attribute Bases | Searched Results |
|:---------------:|:---------------:|:-----------------:|
| ImageNet-1K   | color, size, shape, habitat, behavior                  | (color, shape) |
| Caltech101    | shape, color, material, function, size                 | (shape,size) |
| Oxford Pets   | loyalty, affection, playfulness, energy, intelligence  | (playfulness, energy) |
| Stanford Cars | design, engine, performance, luxury, color             | (luxury) |
| Flowers-102   | color, flower, habitat, growth, season                 | (color, habitat, growth) |
| Food-101      | flavor, texture, origin, ingredients, preparation      | (flavor, preparation) |
| FGVC Aircraft | design, capacity, range, engines, liveries             | (design, range) |
| SUN-397       | architecture, environment, structure, design, function | (function) |
| DTD           | pattern, texture, color, design, structure             | (pattern, color, design) |
| EuroSAT       | habitat, foliage, infrastructure, terrain, watercourse | (habitat) |
| UCF-101       | precision, coordination, technique, strength, control  | (precision) |

Table 1. Attribute bases and searched results for each dataset.
</details>

<hr/>

(2) Reproduce the whole process on your own.

- Register a ChatGPT service account (We are using [ZhiZengZeng](https://gpt.zhizengzeng.com/#/)) and enter the API Key in `gpt_query.py line 27`. Then run the following code:   
```bash
python gpt_query.py
```    
In this way, you will get five output attributes after running the code.    
(You can change the input prompt in `gpt_query.py line 94` to specify as many attributes as you want.)   

- Enter the five attributes into the variables `ATT1_TEXT`, `ATT2_TEXT`, `ATT3_TEXT`, `ATT4_TEXT` and `ATT5_TEXT` in `scripts/attribute_compute/main.sh`. Then run the attribute search code:
```bash
sh scripts/attribute_compute/main.sh
```
Select the result with the **highest confidence** in the last epoch as our target attribute.

In the following **<Training Logs & Weights>**, we provide the complete attribute searching log on ten datasets for your reference.

<hr/>

### Step II: Prompt Learning with ATPrompt.

Here we take the **CoOp+ATPrompt** method as an example. You can switch to other baseline methods if you want.
(This implementation currently supports CoOp+ATPrompt, CoCoOp+ATPrompt, MaPLe+ATPrompt and DePT+ATPrompt methods.)

**(1) Base-to-Novel Experiments.**

1. The config files for each baseline method are provided in `configs/trainers/`. You can modify the hyperparameters in these config files.

2. Change the `DATA` in `scripts/coop/atprompt_base2new_train.sh line 4` to your current dataset path.

3. Run the following commands to train the model using the ATPrompt method:   

**🚀 Training:**
```bash
# CoOp+ATPrompt, dataset=imagenet
sh scripts/coop/atprompt_base2new_train.sh imagenet

# CoOp+ATPrompt, dataset=caltech101
sh scripts/coop/atprompt_base2new_train.sh caltech101
```
**⚡ Testing:**
```bash
# CoOp+ATPrompt, dataset=caltech101
sh scripts/coop/atprompt_base2new_test.sh caltech101
```

If you don't want to use ATPrompt, you can set `TRAINER.ATPROMPT.USE_ATPROMPT` in `scripts/coop/atprompt_base2new_train.sh line 31` to **False**.   
Or you can run the following command:

```bash
# Vanilla CoOp
sh scripts/coop/vanilla_base2new_train.sh imagenet
```

**(2) Cross-dataset & Domain Generalization Experiments.**

1. Change the `DATA` in `scripts/coop/xd_train.sh line 4` to your current dataset path.

2. Train the model on the source dataset (ImageNet) and select the best-performing model.

```bash
sh scripts/coop/atprompt_xd_train.sh
```

3. After training, evaluate the model on other recognition datasets. For example, the model trained with **seed 1** has the best performance.
So we evaluate its performance like this:

```bash
# Cross-dataset
# dataset=caltech101, seed=1
sh scripts/coop/atprompt_xd_eval.sh caltech101 1

# Domain Generalization
# dataset=imagenet_a, seed=1
sh scripts/coop/atprompt_xd_eval.sh imagenet_a 1
```

In the following part, we provide the complete training log and model weights of **CoOp+ATPrompt** for your reference.


<!-- ### 🔬 Experimental Results

The results are averaged over 3 seeds. Note that due to the limited number of training samples and network parameters, the performance results may fluctuate. If you cannot achieve the reported results, please run more experiments with different seeds. -->

<!-- #### Base-to-Novel Generalization

<details>
<summary>Click to expand "Result Figures".</Summary>
<figure>
<img src="images/exp_results.png" alt="fail" width="100%"">    
<figcaption class="content has-text-left" style="word-break:normal">Table 1: Base-to-novel generalization experiments of five baselines with and without our ATPrompt on 11 recognition datasets.
</figure>
</details> -->

## 📄 Training Logs & Weights

- Attribute Search.  
We provide the complete attribute searching log on ten datasets for your reference.   
[[Github Release](https://github.com/zhengli97/ATPrompt/releases/tag/training-log)]

- Base-to-Novel Generalization (CoOp+ATPrompt).   
We provide the complete training logs and model weights on 11 datasets for your reference.  
[[Github Release](https://github.com/zhengli97/ATPrompt/releases/tag/traininglog_and_weights)]

- Cross-dataset Prompt Learning (CoOp+ATPrompt).  
We provide model weights and training logs trained on the source dataset (ImageNet) under cross-dataset settings.  
[[Github Release](https://github.com/zhengli97/ATPrompt/releases/tag/weights)]

## Detailed Hyperparameters for Reproducing

In this part, we provide implementation details and hyperparameter settings for reproducing CoOp+ATPrompt, CoCoOp+ATPrompt, MaPLe+ATPrompt and DePT+ATPrompt. 

**💡 Important Note: Reproduction with the following settings may deviate or fluctuate from the reported values. This is due to the randomness of the training data partitioning (`oxford_pets.py line 77`). This is normal. We recommend that researchers run more experiments with different seeds to reproduce the corresponding results stably.**

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

The above attributes correspond to the `cfg.TRAINER.ATPROMPT.ATT1_TEXT`, `cfg.TRAINER.ATPROMPT.ATT2_TEXT`, and `cfg.TRAINER.ATPROMPT.ATT3_TEXT` variables in the code.

If you want to experiment with other attribute words, you can change the variable values ​​in the function defined in `train.py line 154`.

### Base-to-Novel Experiments

### CoOp+ATPrompt

In this experiment, keep other hyperparameters unchanged. For datasets including Caltech, OxfordPets, StanfordCars, Flowers, Food101, Aircraft, SUN397, EuroSAT, and UCF101, we specifically set EPO=100, NCTX=2.

For the DTD dataset, set EPO=100, NCTX=4.

For the ImageNet dataset, set EPO=10, NCTX=2.

### CoCoOp+ATPrompt

In this experiment, keep other parameters unchanged. For datasets including ImageNet, Caltech, OxfordPets, Food101, FGVC Aircraft, SUN397, and DTD, we specifically set EPO=10, NCTX=2.

For the UCF-101 dataset, set EPO=10, NCTX=4.

For StanfordCars, Flowers, and EuroSAT datasets, set EPO=10, NCTX=6.

### MaPLe+ATPrompt

In this experiment, keep other parameters unchanged. For datasets including Caltech101, OxfordPets, Flowers, and EuroSAT, we specifically set EPO=10, NCTX=4.

For datasets including ImageNet, StanfordCars, Food101, SUN397, DTD, set EPO=5, NCTX=4.

For the FGVC Aircraft and UCF101 datasets, set EPO=5, NCTX=2.

### DePT+ATPrompt

Note: There is a loss balance hyperparameter w in DePT, which is generally set to 0.7 by default.

In this experiment, other parameters are kept unchanged. For the datasets including Caltech, DTD, EuroSAT, FGVC Aircraft, Food101, Flowers, SUN397, and UCF101, we set EPO=10, NCTX=4, where w=0.6 is set for Caltech and DTD datasets, and w=0.5 is set for UCF101.

For the StanfordCars and OxfordPets datasets, we set EPO=10, NCTX=2, where w=0.6 is set for StanfordCars.

### Cross-dataset & Domain Generalization Experiments

In this experimental setting, the usual practice is to select a model trained on the source dataset and measure its generalization performance on other datasets. There will be performance fluctuations when reproducing it yourself, so we recommend that researchers run several seeds and select the best-performing model for evaluation.

### CoOp+ATPrompt

In this experiment, other parameters are kept unchanged. On the ImageNet-1K source dataset, we set NCTX=4 and EPO=10.

### CoCoOp+ATPrompt

In this experiment, other parameters are kept unchanged. On the ImageNet-1K source dataset, we set NCTX=4 and EPO=5.

### MaPLe+ATPrompt

In this experiment, other parameters are kept unchanged. On the ImageNet-1K source dataset, we set NCTX=4 and EPO=2.

## 论文结果复现指引 (中文版)

在这一部分，我们会提供用于复现CoOp+ATPrompt，CoCoOp+ATPrompt, MaPLe+ATPrompt的实现细节和超参数设定。(DePT和PromptKD的实验因为代码还没整合进来，暂时没有提供，DePT和PromptKD由于是用的不一样的实现框架，所以需要一定的时间进行迁移)。

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

### Base-to-Novel实验

### CoOp+ATPrompt

在该实验中，保持其他超参数不变，对于Caltech, OxfordPets, StanfordCars, Flowers, Food101, Aircraft, SUN397, EuroSAT, UCF101，我们特别地设定EPO=100, NCTX=2。

对于DTD, 设定EPO=100, NCTX=4。

对于ImageNet, 设定EPO=10, NCTX=2。

### CoCoOp+ATPrompt

在该实验中，保持其他参数不变，对于ImageNet, Caltech, OxfordPets, Food101, FGVC Aircraft, SUN397, DTD 设定EPO=10, NCTX=2。

对于UCF-101, 设定EPO=10, NCTX=4。

对于StanfordCars, Flowers, EuroSAT, 设定EPO=10, NCTX=6。

### MaPLe+ATPrompt

在该实验中，保持其他参数不变，对于Caltech101, OxfordPets, Flowers, EuroSAT, 设定EPO=10, NCTX=4。

对于ImageNet, StanfordCars, Food101, SUN397, DTD, 设定EPO=5, NCTX=4。

对于FGVC Aircraft, UCF101, 设定EPO=5, NCTX=2。

### DePT+ATPrompt

注：DePT中有一个损失平衡超参数w，一般默认设定为0.7。

在该实验中，保持其他参数不变，对于Caltech, DTD, EuroSAT, FGVC Aircraft, Food101, Flowers, SUN397, UCF101 设定EPO=10, NCTX=4, w=0.7 其中对于Caltech, DTD, 设定EPO=10, NCTX=4, w=0.6， 对于UCF101, 设定EPO=10, NCTX=4, w=0.5。

对于OxfordPets, 设定EPO=10, NCTX=2, w=0.7, 对于StanfordCars，设定EPO=10, NCTX=2, w=0.6。

### Cross-dataset & Domain Generalization 实验

在该实验设定下，因为是选择一个在源域上训练的模型在其他数据集上测泛化，在自己复现的时候会存在性能的波动，所以我们建议研究者多跑几个seed，选择其中较好的模型进行验证评估。

### CoOp+ATPrompt

在该实验中，保持其他参数不变，在ImageNet-1K源数据集上，设定NCTX=4, EPO=10。

### CoCoOp+ATPrompt

在该实验中，保持其他参数不变，在ImageNet-1K源数据集上，设定NCTX=4, EPO=5。

### MaPLe+ATPrompt

在该实验中，保持其他参数不变，在ImageNet-1K源数据集上，设定NCTX=4, EPO=2。
