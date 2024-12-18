# ATPrompt: Textual Prompt Learning with Embedded Attributes

> [**ATPrompt: Textual Prompt Learning with Embedded Attributes**]() <br>
> Zheng Li, Yibing Song, Penghai Zhao, Ming-Ming Cheng, Xiang Li#, Jian Yang#. <br>
> Nankai University, Alibaba DAMO Academy. <br>
> arXiv:2412.09442 <br>
> [[Paper](https://arxiv.org/abs/2412.09442)] [[Project Page](https://zhengli97.github.io/ATPrompt/)] [[Paper Interpretation(TBD)]()] [[中文解读](https://zhuanlan.zhihu.com/p/11787739769)]

<hr/>

### 🔥 News

- 2024.12. We release the official code of ATPrompt and create the project page. The Chinese interpretation of the paper is now available on the [Zhihu forum](https://zhuanlan.zhihu.com/p/11787739769).

<hr />

### 💡 Tips:

1. If you are interested in prompt learning and want to know more about related work, we also maintain a [list of awesome papers](https://github.com/zhengli97/Awesome-Prompt-Adapter-Learning-for-Vision-Language-Models) for your reference.
2. If you are trying to reproduce the results of this implementation on the Stanfordcars dataset, the link to this dataset may be broken and unavailable. We have provided the dataset in [GitHub releases](https://github.com/zhengli97/PromptKD/releases/tag/datasets) for your convenience.

<hr/>

### Abstract

In this work, we introduce an attribute-embedded textual prompt learning method for vision-language models, named ATPrompt.

This method extends the learning space of soft prompts from the original one-dimensional category level to the multi-dimensional attribute level by embedding multiple fixed universal attribute tokens into the learnable soft prompts. 

Guided by these attributes, soft prompts acquire not only category-specific but also attribute-related general representations during training, thereby enhancing the alignment between images and unknown categories compared to the original method.

### Framework

<div style="text-align:center"><img src="images/attribute_compare.png" width="100%"></div>
<figcaption class="content has-text-left"  style="word-break:normal">Figure 1. Architectural comparison among vanilla CLIP, classic prompt learning, and our proposed attribute-embedded prompt learning. </figcaption>

<br>

<div style="text-align:center"><img src="images/shallow_deep_version.png" width="100%"></div>
<figcaption class="content has-text-left" style="word-break:normal">Figure 2. An illustration of the computation process for shallow and deep versions.  </figcaption>


### Highlights

(1). We introduce an attribute-templated prompt learning method for VLMs that utilizes universal attributes to regularize the learning of soft prompts.

(2). We introduce a differentiable attribute search method that learns to determine the appropriate attribute content and quantity.

(3). Both shallow and deep versions of ATPrompt are introduced to achieve compatibility with existing methods.

(4). ATPrompt can be seamlessly intergrated into existing textual-based methods and brings general improvement at a negligible computational cost.


## 🚀 Running

### Preliminary

1. Create the environment and install Dassl.pytorch library. Please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md).

2. Prepare the dataset. Please follow the instructions detailed in [DATASETS.md](docs/DATASETS.md). If you are unable to access the StanfordCars dataset, we have provided the dataset in [[GitHub Release]((https://github.com/zhengli97/PromptKD/releases/tag/datasets))] for your convenience.

3. (Optional) Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder. Comment the `trainers/coop.py line 42` and uncomment the `line 43`.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

### 🚀 Running ATPrompt

#### Step I: Attribute Search (Optional)

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

In the following part, we provide the complete training log on Caltech101 for your reference.

#### Step II: Prompt Learning with ATPrompt.

Here we take the **CoOp+ATPrompt** method as an example. You can switch to other baseline methods if you want.

(1) Base-to-Novel Experiments.

1. The config files for each baseline method are provided in `configs/trainers/`. You can modify the hyper-parameters in these config files.

2. Change the `DATA` in `scripts/coop/base2new_train.sh line 4` to your current dataset path.

3. Run the following commands to train the model using the ATPrompt method:   

**🚀  Training:**
```bash
# CoOp+ATPrompt, dataset=imagenet
sh scripts/coop/atp_base2new_train.sh imagenet

# CoOp+ATPrompt, dataset=caltech101
sh scripts/coop/atp_base2new_train.sh caltech101
```
**⚡ Testing:**
```bash
# CoOp+ATPrompt, dataset=caltech101
sh scripts/coop/atp_base2new_test.sh caltech101
```

If you don't want to use ATPrompt, you can set `TRAINER.ATPROMPT.USE_ATPROMPT` in `scripts/coop/base2new_train.sh line 31` to **False**.   
Or you can run the following command:

```bash
# Vanilla CoOp
sh scripts/coop/vanilla_base2new_train.sh imagenet
```

For more details, please refer to `docs/`.

(2) Cross-dataset & Domain Generalization Experiments.

1. Change the `DATA` in `scripts/coop/xd_train.sh line 4` to your current dataset path.

2. Train the model on the source dataset (ImageNet) and select the best performing model.

```bash
sh scripts/coop/xd_train.sh
```

3. After training, evaluate the model on other recognition datasets. For example, the model trained with **seed 1** has the best performance.
So we evaluate its performance like this:

```bash
# Cross-dataset
# dataset=caltech101, seed=1
sh scripts/coop/xd_eval.sh caltech101 1

# Domain Generalization
# dataset=imagenet_a, seed=1
sh scripts/coop/xd_eval.sh imagenet_a 1
```

In the following part, we provide the complete training log and model weights of **CoOp+ATPrompt** for your reference.


## 🔬 Experimental Results

The results are averaged over 3 seeds. Note that due to the limited number of training samples and network parameters, the performance results may fluctuate. If you cannot achieve the reported results, please run more experiments with different seeds.

### Base-to-Novel Generalization

<details>
<summary>Click to expand "Result Figures".</Summary>
<figure>
<img src="images/exp_results.png" alt="fail" width="100%"">    
<figcaption class="content has-text-left" style="word-break:normal">Table 1: Base-to-novel generalization experiments of five baselines with and without our ATPrompt on 11 recognition datasets. HM: Harmonic Mean. ∆: HM improvement of ATPrompt over previous results. "ATPrompt" is abbreviated as "ATP". Our method achieves consistent average performance improvement over different baselines.
</figure>
</details>

### Cross-dataset Experiments

<details>
<summary>Click to expand "Result Figures".</Summary>
<figure>
<img src="images/exp_results2.png" alt="fail" width="100%"">   
<figcaption class="content has-text-left" style="word-break:normal">Table 2: Cross-dataset generalization experiments of three baselines with and without our ATPrompt on 11 datasets. Our method achieves consistent average performance improvements over three baseline methods.
</figure>
</details>

### Domain Generalization

<details>
<summary>Click to expand "Result Figures".</Summary>
<figure>
<img src="images/exp_results3.png" alt="fail" width="60%"">   
<figcaption class="content has-text-left" style="word-break:normal">Table 3: Domain generalization experiments of three baselines with and without our ATPrompt on 4 datasets. Our method achieves consistent average performance improvement over three baseline methods.
</figure>
</details>

## 📄 Training Logs & Weights

- Attribute Search.  
We provide the complete attribute searching log on the Caltech101 dataset for your reference.   
[[Baidu Cloud(TBD)]()] [[TeraBox(TBD)]()] [[Github Releases(TBD)]()]

- Prompt Learning (CoOp+ATPrompt).  
We provide model weights and training logs trained on the source dataset (ImageNet) under cross-dataset setings.  
[[Baidu Cloud(TBD)]()] [[TeraBox(TBD)]()] [[Github Releases(TBD)]()]


## ✉️ Contact

If you have any questions, you can submit an [issue](https://github.com/zhengli97/ATPrompt/issues) on GitHub, or contact me by email (zhengli97 [at] qq.com).

## ⭐ Citation

If you find our paper or repo helpful for your research, please consider citing the following paper and giving this repo a star. Thank you!

```
@article{li2024atprompt,
  title={ATPrompt: Textual Prompt Learning with Embedded Attributes},
  author={Li, Zheng and Song, Yibing and Zhao, Penghai and Cheng, Ming-Ming and Li, Xiang and Yang, Jian},
  journal={arXiv preprint arXiv:2412.09442},
  year={2024}
}
```

## 👏 Acknowledgements

Our code is based on [PromptSRC](https://github.com/muzairkhattak/PromptSRC), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [KgCoOp](https://github.com/htyao89/KgCoOp), [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code.


