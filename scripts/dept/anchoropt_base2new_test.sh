#!/bin/bash

# custom config
DATA=/root/prompt_dataset
TRAINER=DePT_AnchorOPT

# fix
CFG=vit_b16_ep10_bs4_lr35
SHOTS=16
ANCHOR_LEN=1
GUMBEL_TEMP=1.0

# variable
PROMPT_CE_WEIGHT=1.0
NCTX=4
EPO=10
WEIGHT=0.7

DATASET=#1 # imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101

for SEED in 1 2 3 4 5
do
  DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/CTX_${NCTX}_ACTX_${ANCHOR_LEN}_EPO_${EPO}_CE_${PROMPT_CE_WEIGHT}_W_${WEIGHT}/seed${SEED}
  
  CUDA_VISIBLE_DEVICES=0 python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/DePT/${CFG}.yaml \
      --output-dir ${DIR}/evaluation \
      --model-dir ${DIR} \
      --load-epoch ${EPO} \
      --eval-only \
      DATASET.SUBSAMPLE_CLASSES new \
      TRAINER.LINEAR_PROBE.WEIGHT ${WEIGHT} \
      TRAINER.ANCHOROPT.N_CTX ${NCTX} \
      TRAINER.ANCHOROPT.GUMBEL_TEMP ${GUMBEL_TEMP} \
      TRAINER.ANCHOROPT.ANCHOR_LEN ${ANCHOR_LEN} \
      TRAINER.ANCHOROPT.PROMPT_CE_WEIGHT ${PROMPT_CE_WEIGHT} \
      OPTIM.MAX_EPOCH ${EPO}
done