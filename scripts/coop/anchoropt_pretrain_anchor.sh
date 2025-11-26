#!/bin/bash

DATA=/root/prompt_dataset
TRAINER=CoOp_Pretrain_Anchor

# fix
CFG=vit_b16  # config file
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
ANCHOR_LEN=1
GUMBEL_TEMP=1.0

# variable
ANCHOR_MSE_WEIGHT=1.0
NCTX=4
EPO=20

DATASET=$1 # imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101

for SEED in 1 2 3 4 5
do
  DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/ACTX_${ANCHOR_LEN}_EPO_${EPO}_MSE_${ANCHOR_MSE_WEIGHT}/seed${SEED}

  CUDA_VISIBLE_DEVICES=0 python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CoOp/${CFG}.yaml \
      --output-dir ${DIR} \
      TRAINER.ANCHOROPT.N_CTX ${NCTX} \
      TRAINER.ANCHOROPT.ANCHOR_LEN ${ANCHOR_LEN} \
      TRAINER.ANCHOROPT.ANCHOR_MSE_WEIGHT ${ANCHOR_MSE_WEIGHT} \
      TRAINER.ANCHOROPT.GUMBEL_TEMP ${TEMP} \
      TRAINER.ANCHOROPT.MAX_TEMPLATE_LENGTH 10 \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES base \
      OPTIM.MAX_EPOCH ${EPO}
done
