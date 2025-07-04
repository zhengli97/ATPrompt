import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.coop_atp
import trainers.cocoop
import trainers.cocoop_atp
import trainers.maple
import trainers.maple_atp
import trainers.zsclip

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    # cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    
    cfg.TRAINER.COOP.W = 2.0
    
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    # cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.CTX_INIT = ""  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    
    cfg.TRAINER.ATPROMPT = CN()
    cfg.TRAINER.ATPROMPT.USE_ATPROMPT = False
    cfg.TRAINER.ATPROMPT.N_ATT1 = 4
    cfg.TRAINER.ATPROMPT.N_ATT2 = 4
    cfg.TRAINER.ATPROMPT.N_ATT3 = 4
    cfg.TRAINER.ATPROMPT.ATT_NUM = 0
    cfg.TRAINER.ATPROMPT.ATT1_TEXT = "none"
    cfg.TRAINER.ATPROMPT.ATT2_TEXT = "none"
    cfg.TRAINER.ATPROMPT.ATT3_TEXT = "none"
    
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.TRAINER.IVLP.IMG_WEIGHT = 0.5  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)


def choose_attribute_for_atprompt(cfg):
    if cfg.TRAINER.ATPROMPT.USE_ATPROMPT:
        print("Training models with ATPrompt.")
        if cfg.DATASET.NAME == "ImageNet" or cfg.DATASET.NAME == "ImageNetSketch" or cfg.DATASET.NAME == "ImageNetA" or cfg.DATASET.NAME == "ImageNetR" or cfg.DATASET.NAME == "ImageNetV2":
            cfg.TRAINER.ATPROMPT.ATT_NUM=2
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="color"
            cfg.TRAINER.ATPROMPT.ATT2_TEXT="shape"

        elif cfg.DATASET.NAME ==  "Caltech101":
            cfg.TRAINER.ATPROMPT.ATT_NUM=2
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="shape"
            cfg.TRAINER.ATPROMPT.ATT2_TEXT="size"

        elif cfg.DATASET.NAME ==  "OxfordPets":
            cfg.TRAINER.ATPROMPT.ATT_NUM=2
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="playfulness"
            cfg.TRAINER.ATPROMPT.ATT2_TEXT="energy"

        elif cfg.DATASET.NAME ==  "StanfordCars":
            cfg.TRAINER.ATPROMPT.ATT_NUM=1
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="luxury"

        elif cfg.DATASET.NAME == "OxfordFlowers":
            cfg.TRAINER.ATPROMPT.ATT_NUM=3
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="color"
            cfg.TRAINER.ATPROMPT.ATT2_TEXT="habitat"
            cfg.TRAINER.ATPROMPT.ATT3_TEXT="growth"

        elif cfg.DATASET.NAME == "Food101" :
            cfg.TRAINER.ATPROMPT.ATT_NUM=2
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="flavor"
            cfg.TRAINER.ATPROMPT.ATT2_TEXT="preparation"

        elif cfg.DATASET.NAME == "FGVCAircraft":
            cfg.TRAINER.ATPROMPT.ATT_NUM=2
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="design"
            cfg.TRAINER.ATPROMPT.ATT2_TEXT="range"
            
        elif cfg.DATASET.NAME == "SUN397":
            cfg.TRAINER.ATPROMPT.ATT_NUM=1
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="function"

        elif cfg.DATASET.NAME == "DescribableTextures":
            cfg.TRAINER.ATPROMPT.ATT_NUM=3
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="pattern"
            cfg.TRAINER.ATPROMPT.ATT2_TEXT="color"
            cfg.TRAINER.ATPROMPT.ATT3_TEXT="design"

        elif cfg.DATASET.NAME == "EuroSAT":
            cfg.TRAINER.ATPROMPT.ATT_NUM=1
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="habitat"

        elif cfg.DATASET.NAME == "UCF101":
            cfg.TRAINER.ATPROMPT.ATT_NUM=1
            cfg.TRAINER.ATPROMPT.ATT1_TEXT="precision"

        else:
            print("You are choosing the wrong dataset!")
    else:
        print("Training models with vanilla method.")
    
    print(f"Current dataset is: {cfg.DATASET.NAME}.")

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. Choose Attributes
    choose_attribute_for_atprompt(cfg)
 
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)

