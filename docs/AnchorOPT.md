# How to reproduce AnchorOPT

## Preliminary

1. Create the environment and install Dassl.pytorch library. Please follow the instructions detailed in [[INSTALL.md](INSTALL.md)].

2. Prepare the dataset. Please follow the instructions detailed in [[DATASETS.md](DATASETS.md)]. For your download convenience, we have provided 14 datasets (excluding ImageNet-1K) in the Huggingface platform. [[HuggingFace_Download_Links](https://huggingface.co/zhengli97/prompt_learning_dataset)]

3. Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder. Comment the `trainers/coop.py line 42` and uncomment the `line 43`.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

## 🚀 Running AnchorOPT

### Step I: Pretrain Anchor Token (Optional)


### Step II: Prompt Learning with Dynamic Anchor Token


