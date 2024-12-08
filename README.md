# ATPrompt: Textual Prompt Learning with Embedded Attributes


> [**ATPrompt: Textual Prompt Learning with Embedded Attributes**]() <br>
> Zheng Li, Yibing Song, Penghai Zhao, Ming-Ming Cheng, Xiang Li#, Jian Yang#. <br>
> Nankai University, Alibaba DAMO Academy. <br>
> [[Paper(TBD)]()] [[Project Page(TBD)]()] [[Paper Interpretation(TBD)]()] [[中文解读(TBD)]()]

<hr/>

### Abstract

In this work, we introduce an attribute-embedded textual prompt learning method for vision-language models, named ATPrompt.

### Framework

<div style="text-align:center"><img src="images/attribute_compare.png" width="100%"></div>
<figcaption class="content has-text-left"  style="word-break:normal">Figure 1. Architectural comparison among vanilla CLIP, classic prompt learning, and our proposed attribute-embedded prompt learning. <strong>(a)</strong> Vanilla CLIP employs a hand-crafted text template, “a photo of a {classname}”, as input to the text encoder. <strong>(b)</strong> Classical prompt learning proposes a new text form that concatenates multiple learnable soft tokens with class tokens, replacing the manually designed hard template. <strong>(c)</strong> Our ATPrompt embeds multiple fixed attribute tokens into the set of soft tokens, transforming the original form into an attribute-class mixed form for prompt learning. </figcaption>

### Highlights

(1). We introduce an attribute-templated prompt learning method for VLMs that utilizes universal attributes to regularize the learning of soft prompts.

(2). We introduce a differentiable attribute search method that learns to determine the appropriate attribute content and quantity.

(3). Both shallow and deep versions of ATprompt are introduced to achieve compatibility with existing methods.

(4). ATPrompt can be seamlessly intergrated into existing textual-based methods and brings general improvement at a negligible computational cost.

## Experimental Results

Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.

### Base-to-Novel

<figure>
<img src="images/exp_results.png" alt="fail" width="100%"">
<figcaption class="content has-text-left" style="word-break:normal">Table 1: Base-to-novel generalization experiments of five baselines with and without our ATPrompt on 11 recognition datasets. HM: Harmonic Mean. ∆: HM improvement of ATPrompt over previous results. "ATPrompt" is abbreviated as "ATP". Our method achieves consistent average performance improvement over different baselines.
</figure>

### Cross-dataset

<figure>
<img src="images/exp_results2.png" alt="fail" width="100%"">
<figcaption class="content has-text-left" style="word-break:normal">Table 2: Cross-dataset generalization experiments of three baselines with and without our ATPrompt on 11 datasets. Our method achieves consistent average performance improvements over three baseline methods.
</figure>


## Running

### Preliminary

1. Create the environment and install Dassl.pytorch library. Please follow the instruction detailed in [INSTALL.md](docs/INSTALL.md).

2. Prepare the dataset. Please follow the instructions detailed in [DATASETS.md](docs/DATASETS.md).

<!-- 3. Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder.   -->
<!-- [[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)] -->

### Running ATPrompt

#### Step I: Attribute Search 

**Two Options:**

**(1) Directly use our results.**

Here we provide the five attribute bases obtained by querying the LLM (GPT-4o) and the final result after the differentiable attribute search. You can directly use our searched results for training.

Please expand the list below to see the results:   
<details>
<summary>Attribute Lists</Summary>

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


**(2) Reproduce the whole process on your own.**

- Register a ChatGPT service account and enter the API Key in `gpt_query.py line 27`. Run the code:   
```bash
python gpt_query.py
```    
In this way, you will get five output attributes after running the code.    
(You can change the input prompt in `gpt_query.py line 94` to specify as many attributes as you need.)   

- Enter the five attributes into the variables `ATT1_TEXT`, `ATT2_TEXT`, `ATT3_TEXT`, `ATT4_TEXT` and `ATT5_TEXT` in `scripts/attribute_compute/main.sh`. Then run the attribute search code:
```bash
sh scripts/attribute_compute/main.sh
```
We select the result with the highest confidence in the last epoch as our target attribute.

In the following **Model Zoo** part, we provide the complete training log on Caltech101 for your reference.

#### Step II: Prompt Learning

(1) Base-to-Novel Experiments.

<!-- 1. The config files for each baseline method are provided in `configs/trainers/`. You can modify the hyper-parameters in these config files according to your needs.

2.  -->

(2) Cross-dataset Experiments.



(3) Domain Generalization Experiments.



## Model Zoo

- Attribute Search.

We provide the complete attribute searching log on the Caltech101 dataset for your reference.
[[Baidu Cloud]()] [[TeraBox]()] [[Github Releases]()]

- Prompt Learning.

We provide model weights and training logs trained on the source dataset (ImageNet) under cross-dataset setings. 
[[Baidu Cloud]()] [[TeraBox]()] [[Github Releases]()]


## Contact

If you have any questions, you can submit an [issue](https://github.com/zhengli97/ATPrompt/issues) on GitHub, or contact me by email (zhengli97 [at] qq.com).

## Acknowledgements

Our code is based on [PromptSRC](https://github.com/muzairkhattak/PromptSRC), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [KgCoOp](https://github.com/htyao89/KgCoOp), [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code.


