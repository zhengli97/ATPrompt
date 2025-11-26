# Reproduction Guide for AnchorOPT

## 🔧 Experimental Setup

1. Environment Configuration: Establish the computational environment and install the Dassl.pytorch library following the detailed instructions in [[INSTALL.md](INSTALL.md)].

2. Dataset Preparation: Prepare the required datasets as specified in [[DATASETS.md](DATASETS.md)]. For research convenience, we provide 14 benchmark datasets (excluding ImageNet-1K) via the Huggingface platform. [[HuggingFace_Download_Links](https://huggingface.co/zhengli97/prompt_learning_dataset)]

3. Pretrained CLIP model Acquisition: Download the original ViT-B/16 CLIP model weight from the official OpenAI website. Place these models in the `./clip` directory. Comment the `trainers/coop.py line 42` and uncomment the `line 43`.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

## 🚀 Running AnchorOPT

### Stage I: Anchor Token Pretraining (Two options)

**Option 1: Utilization of Provided Pretrained Weights.**

The `anchor_weights` directory contains pretrained anchor token weights alongside comprehensive training logs. These resources can be directly loaded for model training (refer to lines 255-289 in `coop_anchoropt.py` for implementation details).

**Option 2: Custom Anchor Token Training.**

To train custom anchor tokens, execute the following commands:

**🚀 Training:**
```
# Caltech101 dataset
bash scripts/coop/anchoropt_pretrain_anchor.sh caltech101
```
**⚡ Evaluation:**
```
# for Caltech101 dataset
bash scripts/coop/anchoropt_eval_anchor.sh caltech101
```

Following training completion, select the optimal model weights and store them in the `anchor_weights` directory. Hyperparameter specifications are detailed in subsequent sections.

### Stage II: Dynamic Anchor Token Prompt Learning

**Baseline Method Configuration**
<!-- **Here we take the CoOp+AnchorOPT method as an example.**  -->

This implementation demonstrates the CoOp+AnchorOPT methodology, though alternative baseline approaches (CoCoOp, MaPLe, DePT, ATPrompt) are equally supported.

<!-- You can switch to other baseline methods if you want. This repo currently supports CoOp, CoCoOp, MaPLe, DePT, ATPrompt and AnchorOPT. -->

**(1) Base-to-Novel Experiments.**

1. Modify hyperparameters within the configuration files located in `configs/trainers/`.

2. Update the `DATA` variable in `scripts/coop/anchoropt_base2new_train.sh line 3` to reflect your current dataset path.

3. Run the following training commands:

**🚀 Training:**
```
# CoOp+AnchorOPT, dataset=caltech101
bash scripts/coop/anchoropt_base2new_train.sh caltech101

# CoCoOp+AnchorOPT, dataset=caltehc101
bash scripts/cocoop/anchoropt_base2new_train.sh caltech101
```

**⚡ Evaluation:**
```
# CoOp+AnchorOPT, dataset=caltech101
bash scripts/coop/anchoropt_base2new_test.sh caltech101

# CoCoOp+AnchorOPT, dataset=caltehc101
bash scripts/cocoop/anchoropt_base2new_test.sh caltech101
```

**(2) Cross-Dataset & Domain Generalization Experiments.**

1. Configure the `DATA` variable in `scripts/coop/xd_train.sh line 4` with your dataset path.

2. Train the model on the source dataset (ImageNet) and identify the best-performing model.

```bash
sh scripts/coop/anchoropt_xd_train.sh
```

3. Evaluate the selected model (e.g., seed 1 demonstrating optimal performance) across target domains:

```bash
# Cross-dataset
# dataset=caltech101, seed=1
sh scripts/coop/anchoropt_xd_eval.sh caltech101 1

# Domain Generalization
# dataset=imagenet_a, seed=1
sh scripts/coop/anchoropt_xd_eval.sh imagenet_a 1
```

## 📄 Training Logs & Weights (TBD)

- Anchor Token Pretraining.

Detailed training logs and weights are provided in the `anchor_weights` folder.

- Base-to-Novel Generalization. (TBD)

- Cross-dataset Generalization. (TBD)


## 🔬 Detailed Hyperparameter Setting

To optimize performance, we systematically tuned hyperparameters when integrating AnchorOPT with baseline methods. The following tables detail the configuration parameters for reproducing results across four frameworks: CoOp+AnchorOPT, CoCoOp+AnchorOPT, MaPLe+AnchorOPT, and DePT+AnchorOPT. The anchor token length was fixed at 1 for all experiments.

**💡 Critical Reproducibility Note: Results obtained with these settings may exhibit minor variations from reported values due to stochastic data partitioning (see `oxford_pets.py, line 77`). Such fluctuations are expected in randomized experimental protocols. We recommend conducting multiple trials with different random seeds to achieve stable reproduction of results.**

Abbreviations and their corresponding meanings:  
- NCTX: Soft token length
- EPOCH: Training epochs
- CE: Cross-entropy loss weight
- KD: KL divergence loss weight
- KD_Temp: Distillation temperature parameter

**Stage I: Anchor Pretraining**
| Dataset | EPOCH | MSE   |
| :---:   | :---: | :---: |
| ImageNet      | 10  | 1000.0 | 
| Caltech101    | 100 | 1000.0 | 
| Oxford Pets   | 100 | 1000.0 | 
| Stanford Cars | 40  | 100.0  | 64.56 75.266
| Flowers102    | 100 | 1000.0 | 
| Food101       | 20  | 1000.0 |
| FGVC Aircraft | 20  | 1000.0 |
| SUN397        | 20  | 1000.0 |
| DTD           | 100 | 1000.0 |
| EuroSAT       | 100 | 1000.0 |
| UCF101        | 100 | 1000.0 |

**Stage II: CoOp+AnchorOPT:**
| Dataset | NCTX  | EPOCH | CE    | KD    | KD_Temp |
| :---:   | :---: | :---: | :---: | :---: | :---:   |
| ImageNet      | 6 | 20  | 100.0 | 1000.0 | 1.0 | 
| Caltech101    | 4 | 100 | 100.0 | 1000.0 | 2.0 |
| Oxford Pets   | 4 | 100 | 100.0 | 1000.0 | 1.0 |
| Stanford Cars | 8 | 100 | 100.0 | 100.0  | 4.0 |
| Flowers102    | 4 | 100 | 100.0 | 100.0  | 2.0 |
| Food101       | 4 | 20  | 100.0 | 1000.0 | 4.0 |
| FGVC Aircraft | 4 | 100 | 100.0 | 1.0    | 4.0 |
| SUN397        | 8 | 20  | 100.0 | 1.0    | 2.0 |
| DTD           | 4 | 100 | 100.0 | 10.0   | 4.0 |
| EuroSAT       | 4 | 100 | 1.0   | 0.0    | 1.0 |
| UCF101        | 8 | 100 | 100.0 | 1000.0 | 1.0 |


**Stage II: CoCoOp+AnchorOPT:**
| Dataset | NCTX  | EPOCH | CE    | KD    | KD_Temp |
| :---:   | :---: | :---: | :---: | :---: | :---:   |
| ImageNet      | 4 | 10 | 100.0 | 100.0  | 4.0 | 
| Caltech101    | 4 | 20 | 10.0  | 1000.0 | 4.0 | 
| Oxford Pets   | 4 | 20 | 10.0  | 100.0  | 2.0 | 
| Stanford Cars | 4 | 20 | 100.0 | 100.0  | 4.0 | 
| Flowers102    | 4 | 20 | 10.0  | 100.0  | 4.0 |
| Food101       | 4 | 20 | 10.0  | 100.0  | 4.0 |
| FGVC Aircraft | 4 | 20 | 10.0  | 100.0  | 0.5 |
| SUN397        | 8 | 10 | 10.0  | 1000.0 | 2.0 |
| DTD           | 4 | 20 | 10.0  | 100.0  | 4.0 |
| EuroSAT       | 4 | 20 | 100.0 | 100.0  | 0.5 |
| UCF101        | 4 | 20 | 10.0  | 100.0  | 0.1 |

**Stage II: MaPLe+AnchorOPT:**
| Dataset | NCTX  | EPOCH | CE    | KD    | KD_Temp |
| :---:   | :---: | :---: | :---: | :---: | :---:   |
| ImageNet      | 8 | 5  | 1.0  | 100.0  | 1.0 | 
| Caltech101    | 8 | 10 | 10.0 | 1000.0 | 2.0 | 
| Oxford Pets   | 8 | 10 | 1.0  | 10.0   | 0.5 | 
| Stanford Cars | 4 | 20 | 10.0 | 0.0    | 0.0 | 
| Flowers102    | 4 | 20 | 10.0 | 100.0  | 2.0 |
| Food101       | 8 | 10 | 1.0  | 100.0  | 1.0 |
| FGVC Aircraft | 4 | 10 | 1.0  | 10.0   | 4.0 |
| SUN397        | 4 | 20 | 10.0 | 1000.0 | 4.0 |
| DTD           | 4 | 20 | 1.0  | 10.0   | 4.0 |
| EuroSAT       | 4 | 20 | 1.0  | 1.0    | 2.0 |
| UCF101        | 8 | 10 | 1.0  | 1.0    | 2.0 |

**Stage II: DePT+AnchorOPT:**
| Dataset | NCTX  | EPOCH | CE    | W     |
| :---:   | :---: | :---: | :---: | :---: | 
| ImageNet      | 4 | 10 | 10.0 | 0.5 | 
| Caltech101    | 4 | 10 | 10.0 | 0.5 | 
| Oxford Pets   | 4 | 10 | 10.0 | 0.5 | 
| Stanford Cars | 8 | 20 | 10.0 | 0.6 |
| Flowers102    | 8 | 10 | 10.0 | 0.9 |
| Food101       | 4 | 10 | 10.0 | 0.5 |
| FGVC Aircraft | 4 | 20 | 10.0 | 0.6 |
| SUN397        | 8 | 10 | 10.0 | 0.6 |
| DTD           | 8 | 10 | 10.0 | 0.8 |
| EuroSAT       | 8 | 20 | 10.0 | 0.9 |
| UCF101        | 8 | 10 | 10.0 | 0.8 |

