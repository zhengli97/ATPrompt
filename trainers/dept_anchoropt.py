# CoOp w/ DePT
import os
import os.path as osp
import torch
import torch.nn.functional as F
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from tqdm import tqdm
import sys, os

from .coop_anchoropt import CoOp_AnchorOPT, load_clip_to_cpu
from .coop_anchoropt import CustomCLIP as CustomCLIP_
from dassl.optim import build_optimizer

class FiLM(nn.Module):
    def __init__(self, 
                 dim, 
                 bias=True, 
                 use_sigmoid=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.has_bias = bias
        self.use_sigmoid = use_sigmoid
    
    def forward(self, x):
        scale = self.scale.unsqueeze(0).type(x.dtype)
        bias = self.bias.unsqueeze(0).type(x.dtype) if self.has_bias else None
        
        x = scale * x
        if bias is not None:
            x = x + bias
        
        if self.use_sigmoid:
            return x.sigmoid()
        
        return x

class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.dataset = cfg.DATASET.NAME
        self.lp_cfg = cfg.TRAINER.LINEAR_PROBE
        self.film_cfg = cfg.TRAINER.FILM

        clip_dim = clip_model.text_projection.size(1)
        
        film_cfg = self.film_cfg

        if film_cfg.LINEAR_PROBE:
            # cwT module
            self.film_lp_img = FiLM(clip_dim)
            self.film_lp_text = FiLM(clip_dim)
        
        # for base to new, base classes will be 'base'
        # for cross dataset, classes from ImageNet will be 'base'
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            assert self.lp_cfg.TYPE in ['similarity', 'linear']

            # linear classifier
            if self.lp_cfg.TYPE == 'similarity':
                self.linear_probe_proj = nn.Identity()
            elif self.lp_cfg.TYPE == 'linear':
                self.linear_probe_proj = nn.Linear(clip_dim, len(classnames)).type(self.dtype)
        else:
            self.linear_probe_proj = nn.Identity()
        
        self.prompt_ce_weight = cfg.TRAINER.ANCHOROPT.PROMPT_CE_WEIGHT
        
    def forward(self, img, labels=None):
        
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            return self._forward_base(img, labels)
        else:
            return self._forward_new(img)

    def _forward_base(self, img, labels=None):
        """ forward function for base classes """
        text_feats, img_feats, anchor_text_feats = self._forward_feats(img)
        
        # forward similartiy and linear logits
        logits, logits_anchor = self._forward_logits_similarity(text_feats, img_feats, anchor_text_feats)
        logits_lp, labels_lp = self._forward_logits_linear_probe(text_feats, img_feats, labels)
        
        if self.normal_prompt_learner.training:
            # while training, return loss of both logits
            return self._loss(logits, labels, logits_lp, labels_lp)
        
        if not self.lp_cfg.TEST_TIME_FUSION:
            return logits_lp

        # while inference, fusion both logits and return
        lp_weight = self.lp_cfg.WEIGHT
        logits = (1 - lp_weight) * logits + lp_weight * logits_lp
        
        return logits, logits_anchor
    
    def _forward_new(self, img):
        """ forward function for new classes """
        assert not self.normal_prompt_learner.training
        
        # for new classes, only forward similarity logits
        text_feats, img_feats, anchor_text_feats = self._forward_feats(img)
        logits, logits_anchor = self._forward_logits_similarity(text_feats, img_feats, anchor_text_feats)
        
        return logits, logits_anchor
    
    def _forward_feats(self, img):
        
        with torch.no_grad():
            anchor_prompts, anchor_1 = self.anchor_prompt_learner()
            anchor_tokenized_prompts = self.anchor_tokenized_prompts
            anchor_text_features = self.text_encoder(anchor_prompts, anchor_tokenized_prompts)
            anchor_text_feats = anchor_text_features / anchor_text_features.norm(dim=-1, keepdim=True)

        prompts = self.normal_prompt_learner(anchor_1.detach())

        tokenized_prompts = self.normal_tokenized_prompts
        text_feats = self.text_encoder(prompts, tokenized_prompts)
        img_feats = self.image_encoder(img.type(self.dtype))

        return text_feats, img_feats, anchor_text_feats
    
    def _forward_logits_similarity(self, text_feats, img_feats, anchor_text_feats):
        # normalize and calcute cosine similarity
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * img_feats @ text_feats.t()
        logits_anchor = logit_scale * img_feats @ anchor_text_feats.t()
    
        return logits, logits_anchor
    
    def _forward_logits_linear_probe(self, text_feats, img_feats, labels):
        # cwT module
        if self.film_cfg.LINEAR_PROBE:
            text_feats = self.film_lp_text(text_feats)
            img_feats = self.film_lp_img(img_feats)

        # while new head is similarity head, use similarity forward function
        if self.lp_cfg.TYPE == 'similarity':
            return self._forward_logits_similarity(text_feats, img_feats), labels
 
        if labels is None:
            # while inference, forward image features only
            all_feats = img_feats
            all_labels = labels
        else:
            # while training, image features and text features will be concated to train classifier
            text_feats = text_feats[labels]
            all_feats = torch.cat([text_feats, img_feats])
            all_labels = torch.cat([labels, labels])

        all_logits = self.linear_probe_proj(all_feats)
        return all_logits, all_labels
    
    def _loss(self, logits, labels, logits_lp, labels_lp):
        # calculate similarity loss and linear loss
        loss_cls = F.cross_entropy(logits, labels)
        loss_cls_lp = F.cross_entropy(logits_lp, labels_lp)

        lp_weight = self.lp_cfg.WEIGHT
        loss = self.prompt_ce_weight * ((1 - lp_weight) * loss_cls + lp_weight * loss_cls_lp)

        return loss
    

@TRAINER_REGISTRY.register()
class DePT_AnchorOPT(CoOp_AnchorOPT):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            optim.zero_grad()

            loss = model(image, label)
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

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = cfg.TRAINER.NAMES_TO_UPDATE

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
                
            param.requires_grad_(update)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print(f"Parameters to be updated: {list(sorted(enabled))}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self._models["normal_prompt_learner"] = self.model.normal_prompt_learner
        self._scheds["normal_prompt_learner"] = self.sched
        self._optims["normal_prompt_learner"] = self.optim

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

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
            if epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)

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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            if "token_and" in state_dict:
                del state_dict["token_and"]
            if "token_of" in state_dict:
                del state_dict["token_of"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            # for some dataset in domain generalization, number of target classes is different from number of source classes
            # thus a mapping must be created to preserve the required class weights
            if self.cfg.DATASET.NAME in ['ImageNetA', 'ImageNetR']:
                from datasets.imagenet import ImageNet
                from dassl.utils import listdir_nohidden

                # read classes from source dataset
                dataset = self.dm.dataset
                text_file = osp.join(dataset.dataset_dir, "classnames.txt")
                all_folders = ImageNet.read_classnames(text_file).keys()

                # read classes from target dataset
                TO_BE_IGNORED = ["README.txt"]
                folders = listdir_nohidden(dataset.image_dir, sort=True)
                folders = [f for f in folders if f not in TO_BE_IGNORED]

                # find that which class from target dataset is in source dataset
                is_reserves = [f in folders for f in all_folders]

                # only reserve required class weights
                print(f'State dict is CLIPPED to match the shape of target dataset {self.cfg.DATASET.NAME}!')
                state_dict['linear_probe_proj.weight'] = state_dict['linear_probe_proj.weight'][is_reserves]
                state_dict['linear_probe_proj.bias'] = state_dict['linear_probe_proj.bias'][is_reserves]
                
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
        
        pos_params = self.model.normal_prompt_learner.ctx_pos_params
        permutation = F.gumbel_softmax(pos_params, tau=self.cfg.TRAINER.ANCHOROPT.GUMBEL_TEMP, hard=True)  # 生成置换矩阵 16,16
        print(f'matrix is {permutation}')

        # if is_final:
        sys.stdout.flush()
        sys.stdout.close()
        os._exit(0)

        return list(results.values())[0]
