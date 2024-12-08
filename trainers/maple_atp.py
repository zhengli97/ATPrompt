import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip import model_atp
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if cfg.TRAINER.ATPROMPT.ATT_NUM==0:
        design_details = {"trainer": 'MaPLe',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "maple_length": cfg.TRAINER.MAPLE.N_CTX
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        design_details = {"trainer": 'MaPLe',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "maple_length": cfg.TRAINER.MAPLE.N_CTX,
                        "atprompt_num": cfg.TRAINER.ATPROMPT.ATT_NUM
        }
        model = model_atp.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
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
        
        self.ctx = nn.Parameter(ctx_vectors)

        self.use_atp = cfg.TRAINER.ATPROMPT.USE_ATPROMPT
        self.atp_num = cfg.TRAINER.ATPROMPT.ATT_NUM

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
  
        if self.use_atp:
            print("USE ATPROPMT-ING 1")
            n_att1 = cfg.TRAINER.ATPROMPT.N_ATT1
            att1_text = cfg.TRAINER.ATPROMPT.ATT1_TEXT
            n_att2 = cfg.TRAINER.ATPROMPT.N_ATT2
            att2_text = cfg.TRAINER.ATPROMPT.ATT2_TEXT
            n_att3 = cfg.TRAINER.ATPROMPT.N_ATT3
            att3_text = cfg.TRAINER.ATPROMPT.ATT3_TEXT
            
            att_vectors_1 = torch.empty(n_att1, ctx_dim, dtype=dtype)
            att_vectors_2 = torch.empty(n_att2, ctx_dim, dtype=dtype)
            att_vectors_3 = torch.empty(n_att3, ctx_dim, dtype=dtype)
            
            nn.init.normal_(att_vectors_1, std=0.01)
            prefix1 = " ".join(["X"] * n_att1)
            nn.init.normal_(att_vectors_2, std=0.01)
            prefix2 = " ".join(["X"] * n_att2)
            nn.init.normal_(att_vectors_3, std=0.01)
            prefix3 = " ".join(["X"] * n_att3)

            self.ctx_att1 = nn.Parameter(att_vectors_1)
            self.ctx_att2 = nn.Parameter(att_vectors_2)
            self.ctx_att3 = nn.Parameter(att_vectors_3)
            
            if self.atp_num == 1:
                prompts = [prefix1 + " " + att1_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 2:
                prompts = [prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 3:
                prompts = [prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prefix3 + " " + att3_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            else:
                print("wrong parameter.")
        else:
            prompts = [prompt_prefix + " " + name + "." for name in classnames] 
        print(prompts)
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])

        if self.use_atp:
            print("USE ATPROPMT-ING 2")
            if self.atp_num == 1:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : 1+n_att1+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_ctx :, :])

            elif self.atp_num == 2:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : 1+n_att1+1, :])
                self.register_buffer("token_middle2", embedding[:, 1+n_att1+1+n_att2 : 1+n_att1+1+n_att2+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_att2+1+n_ctx :, :])

            elif self.atp_num == 3:
                self.register_buffer("token_middle1", embedding[:, 1+n_att1 : 1+n_att1+1, :])
                self.register_buffer("token_middle2", embedding[:, 1+n_att1+1+n_att2 : 1+n_att1+1+n_att2+1, :])
                self.register_buffer("token_middle3", embedding[:, 1+n_att1+1+n_att2+1+n_att3 : 1+n_att1+1+n_att2+1+n_att3+1, :])
                self.register_buffer("token_suffix", embedding[:, 1+n_att1+1+n_att2+1+n_att3+1+n_ctx:, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
            
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, prefix, ctx, suffix, ctx_att1=None, middle_attribute1=None, ctx_att2=None, middle_attribute2=None, ctx_att3=None, middle_attribute3=None, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        if self.use_atp:
            if self.atp_num == 1:
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 2:
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx_att2,
                        middle_attribute2,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 3:
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx_att2,
                        middle_attribute2,
                        ctx_att3,
                        middle_attribute3,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
        else:
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix, 
                ],
                dim=1,
            )
            
        return prompts

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
    
        ctx = self.ctx
        if self.use_atp:
            ctx_att1 = self.ctx_att1
            ctx_att2 = self.ctx_att2
            ctx_att3 = self.ctx_att3

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            if self.use_atp:
                ctx_att1 = ctx_att1.unsqueeze(0).expand(self.n_cls, -1, -1)
                ctx_att2 = ctx_att2.unsqueeze(0).expand(self.n_cls, -1, -1)
                ctx_att3 = ctx_att3.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.use_atp:
            if self.atp_num == 1:
                middle_attribute1 = self.token_middle1
                prompts = self.construct_prompts(prefix, ctx, suffix, ctx_att1, middle_attribute1)
            elif self.atp_num == 2:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                prompts = self.construct_prompts(prefix, ctx, suffix, ctx_att1, middle_attribute1, ctx_att2, middle_attribute2)
            elif self.atp_num == 3:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                middle_attribute3 = self.token_middle3
                prompts = self.construct_prompts(prefix, ctx, suffix, ctx_att1, middle_attribute1, ctx_att2, middle_attribute2, ctx_att3, middle_attribute3)
            else:
                raise ValueError
        else:
            prompts = self.construct_prompts(prefix, ctx, suffix)
            
        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision= self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, None, None, None, None)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe_ATP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

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

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

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

            if "prompt_learner.token_middle1" in state_dict:
                del state_dict["prompt_learner.token_middle1"]
            if "prompt_learner.token_middle2" in state_dict:
                del state_dict["prompt_learner.token_middle2"]
            if "prompt_learner.token_middle3" in state_dict:
                del state_dict["prompt_learner.token_middle3"]
        
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


