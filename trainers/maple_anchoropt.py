import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from tqdm import tqdm
import sys, os

from clip import clip
from clip import model_anchoropt
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    # model_path = './clip/ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'MaPLe',
                    "vision_depth": 0,
                    "language_depth": 0, "vision_ctx": 0,
                    "language_ctx": 0,
                    "maple_length": cfg.TRAINER.ANCHOROPT.N_CTX,
                    "attribute_len": cfg.TRAINER.ANCHOROPT.ANCHOR_LEN
    }
    model = model_anchoropt.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_clip_to_cpu_vanilla(cfg):
    model_path = './clip/ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder_maple(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text, permutation):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0, permutation]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TextEncoder_vanilla(nn.Module):
    def __init__(self, clip_model_vanilla):
        super().__init__()
        self.transformer = clip_model_vanilla.transformer
        self.positional_embedding = clip_model_vanilla.positional_embedding
        self.ln_final = clip_model_vanilla.ln_final
        self.text_projection = clip_model_vanilla.text_projection
        self.dtype = clip_model_vanilla.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class AnchorPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = cfg.TRAINER.COOP.N_CTX

        anchor_len = cfg.TRAINER.ANCHOROPT.ANCHOR_LEN

        # ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prompt_prefix = " ".join(["X"] * anchor_len) + " of"

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {anchor_len}")

        anchor_ctx_1 = torch.empty(anchor_len, ctx_dim, dtype=dtype)
        nn.init.normal_(anchor_ctx_1, std=0.02)
        self.anchor_ctx_1 = nn.Parameter(anchor_ctx_1, requires_grad=False)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # 具体的每个classname位置 对应的是英文名的长度
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_of", embedding[:, 1+anchor_len: 1+anchor_len+1, :])
        self.register_buffer("token_suffix", embedding[:, 1+anchor_len+1:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
    def forward(self):
        anchor_ctx_1 = self.anchor_ctx_1

        if anchor_ctx_1.dim() == 2:
            anchor_ctx_1 = anchor_ctx_1.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        token_of = self.token_of
        
        new_ctx = torch.cat([anchor_ctx_1, token_of], dim=1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim) [1, 512]
                new_ctx,     # (n_cls, n_ctx, dim) [16, 512]
                suffix,  # (n_cls, *, dim) [17:77, 512]
            ],
            dim=1,
        )

        return prompts, anchor_ctx_1


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.ANCHOROPT.N_CTX

        anchor_len = cfg.TRAINER.ANCHOROPT.ANCHOR_LEN
        temp = cfg.TRAINER.ANCHOROPT.GUMBEL_TEMP

        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if n_ctx <= 4:
        if False:
            # use given words to initialize context vectors
            ctx_init = "a photo of a"
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        
        pos_params = torch.randn(n_ctx + anchor_len, n_ctx + anchor_len)
        self.ctx_pos_params = nn.Parameter(pos_params, requires_grad=True) # 16, 16

        self.ctx = nn.Parameter(ctx_vectors)

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])

        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
  
        prompts = [prompt_prefix + " " + name + "." for name in classnames] 
        print(prompts)
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx : -anchor_len, :])  # CLS, EOS
            
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.temp = temp

    def forward(self, anchor_1):
        ctx = self.ctx
        temp = self.temp

        anchor_1 = anchor_1.detach()

        pos_params = self.ctx_pos_params.half().cuda()
        permutation = F.gumbel_softmax(pos_params, tau=temp, hard=True)  # 生成置换矩阵 16,16

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            permutation = permutation.unsqueeze(0).expand(self.n_cls, -1, -1)  # 50,16,16
        
        new_ctx = torch.cat([ctx, anchor_1], dim=1)
        new_ctx = torch.matmul(permutation, new_ctx)  # size (n×d) 16,512

        prefix = self.token_prefix
        suffix = self.token_suffix
    
        prompts = torch.cat(
            [
                prefix,
                new_ctx,
                suffix, 
            ],
            dim=1,
        )
            
        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts, permutation   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, clip_model_vanilla):
        super().__init__()

        self.anchor_prompt_learner = AnchorPromptLearner(cfg, classnames, clip_model)

        if cfg.DATASET.NAME == "Caltech101": # 97.93 95.42
            model_path = "anchor_weights/Caltech101/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME ==  "OxfordPets": # 94.31 98.15
            model_path = "anchor_weights/OxfordPets/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME ==  "StanfordCars": # 64.56 75.266
            model_path = "anchor_weights/StanfordCars/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "OxfordFlowers": # 73.41 78.86
            model_path = "anchor_weights/OxfordFlowers/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "Food101" : # 90.04 91.62
            model_path = "anchor_weights/Food101/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "FGVCAircraft": # 28.87 36.65
            model_path = "anchor_weights/FGVCAircraft/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "SUN397": # 74.36 78.37
            model_path = "anchor_weights/SUN397/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "DescribableTextures": # 55.78 62.56
            model_path = "anchor_weights/DTD/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "EuroSAT": # 57.33 80.20 better: 60.76, 78.69
            model_path = "anchor_weights/EuroSAT/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "UCF101": # 75.33 78.15
            model_path = "anchor_weights/UCF101/anchor_prompt_learner/model.pth.tar"

        elif cfg.DATASET.NAME == "ImageNet" or cfg.DATASET.NAME == "ImageNetA" or cfg.DATASET.NAME == "ImageNetSketch" or cfg.DATASET.NAME == "ImageNetV2" or cfg.DATASET.NAME == "ImageNetR": # 74.63 71.02
            model_path = "anchor_weights/ImageNet/anchor_prompt_learner/model.pth.tar"

        else:
            raise ValueError("you pick the wrong dataset name")

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]

        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]
        if "token_of" in state_dict:
            del state_dict["token_of"]

        if "anchor_prompt_learner.token_prefix" in state_dict:
            del state_dict["anchor_prompt_learner.token_prefix"]
        if "anchor_prompt_learner.token_suffix" in state_dict:
            del state_dict["anchor_prompt_learner.token_suffix"]
        if "anchor_prompt_learner.token_of" in state_dict:
            del state_dict["anchor_prompt_learner.token_of"]
        
        print("Loading weights to {} " 'from "{}"'.format("anchor_prompt", model_path))

        self.anchor_prompt_learner.load_state_dict(state_dict, strict=False)
        self.anchor_tokenized_prompts = self.anchor_prompt_learner.tokenized_prompts

        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder_vanilla = clip_model_vanilla.visual
        self.text_encoder_vanilla = TextEncoder_vanilla(clip_model_vanilla)

        self.image_encoder_maple = clip_model.visual
        self.text_encoder_maple = TextEncoder_maple(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):

        logit_scale = self.logit_scale.exp()
        
        with torch.no_grad():
            anchor_prompts, anchor_1 = self.anchor_prompt_learner()
            anchor_tokenized_prompts = self.anchor_tokenized_prompts
            anchor_text_features = self.text_encoder_vanilla(anchor_prompts, anchor_tokenized_prompts)
            anchor_text_features = anchor_text_features / anchor_text_features.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.tokenized_prompts
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, permutation= self.prompt_learner(anchor_1)
        text_features = self.text_encoder_maple(prompts, tokenized_prompts, deep_compound_prompts_text, permutation)

        image_features = self.image_encoder_maple(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, None)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        image_features_vanilla = self.image_encoder_vanilla(image.type(self.dtype))
        image_features_vanilla = image_features_vanilla / image_features_vanilla.norm(dim=-1, keepdim=True)

        logits_anchor = logit_scale * image_features_vanilla @ anchor_text_features.t()

        return logits, logits_anchor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe_AnchorOPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")

        clip_model_maple = load_clip_to_cpu(cfg)

        clip_model_vanilla = load_clip_to_cpu_vanilla(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model_maple.float()
            clip_model_vanilla.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model_maple, clip_model_vanilla)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.prompt_ce_weight = cfg.TRAINER.ANCHOROPT.PROMPT_CE_WEIGHT
        self.kd_temperature = cfg.TRAINER.ANCHOROPT.KD_TEMPERATURE
        self.kd_weight = cfg.TRAINER.ANCHOROPT.KD_WEIGHT

    def forward_backward_prompt(self, image, label, model, optimizer, prec):
        if prec == 'amp':
            with autocast():
                output = model(image)
                loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            logits_normal, logits_anchor = model(image)

            ens = (logits_anchor + logits_normal)/2
            
            kd_loss = F.kl_div(
                F.log_softmax(logits_normal / self.kd_temperature, dim=1),
                F.softmax(ens.detach() / self.kd_temperature, dim=1),
                reduction='sum',
            ) * (self.kd_temperature * self.kd_temperature) / ens.numel()  # 求平均
            
            loss = self.prompt_ce_weight * F.cross_entropy(logits_normal, label) + self.kd_weight * kd_loss

            loss.backward()
            optimizer.step()

        loss_summary = {'loss_val': loss.item()}
        return loss_summary


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        loss_summary={}

        model = self.model
        prec = self.cfg.TRAINER.MAPLE.PREC

        optim_prompt=self.optim

        optim_prompt.zero_grad()

        loss_prompt = self.forward_backward_prompt(image, label, model, optim_prompt, prec)

        loss_summary.update(loss_prompt)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            if "anchor_prompt_learner.token_prefix" in state_dict:
                del state_dict["anchor_prompt_learner.token_prefix"]
            if "anchor_prompt_learner.token_suffix" in state_dict:
                del state_dict["anchor_prompt_learner.token_suffix"]
            if "anchor_prompt_learner.token_of" in state_dict:
                del state_dict["anchor_prompt_learner.token_of"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_and" in state_dict:
                del state_dict["token_and"]

            if "token_of" in state_dict:
                del state_dict["token_of"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    @torch.no_grad()
    def test(self, is_final=False, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        print(f"evaluate on normal prompts")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            
            with torch.no_grad():
                logits_normal, logits_anchor = self.model(image)

            if self.cfg.DATASET.SUBSAMPLE_CLASSES == "base":
                self.evaluator.process(logits_normal, label)
            elif self.cfg.DATASET.SUBSAMPLE_CLASSES == "new":
                logits_ens = (logits_normal+logits_anchor)/2
                self.evaluator.process(logits_ens, label)

        results = self.evaluator.evaluate()
        
        pos_params = self.model.prompt_learner.ctx_pos_params
        permutation = F.gumbel_softmax(pos_params, tau=self.cfg.TRAINER.ANCHOROPT.GUMBEL_TEMP, hard=True)
        print(f'position matrix is {permutation}')

        # if is_final:
        sys.stdout.flush()
        sys.stdout.close()
        os._exit(0)

        return list(results.values())[0]
