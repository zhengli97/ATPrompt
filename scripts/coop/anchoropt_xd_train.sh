#!/bin/bash

DATA=/root/prompt_dataset
TRAINER=CoOp_AnchorOPT

# fix
CFG=vit_b16  # config file
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
ANCHOR_LEN=1
GUMBEL_TEMP=1.0

# variable
PROMPT_CE_WEIGHT=1.0
KD_WEIGHT=1.0
KD_TEMPERATURE=1.0
NCTX=4
EPO=20

DATASET=imagenet

for SEED in 1 2 3 4 5
do 
  DIR=output/${DATASET}/${TRAINER}_cross/${CFG}_${SHOTS}shots/CTX_${NCTX}_ACTX_${ANCHOR_LEN}_EPO_${EPO}_CE_${PROMPT_CE_WEIGHT}_KD_W_${KD_WEIGHT}_T_${KD_TEMPERATURE}/seed${SEED}

  CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CoOp/${CFG}.yaml \
      --output-dir ${DIR} \
      TRAINER.ANCHOROPT.N_CTX ${NCTX} \
      TRAINER.ANCHOROPT.ANCHOR_LEN ${ANCHOR_LEN} \
      TRAINER.ANCHOROPT.PROMPT_CE_WEIGHT ${PROMPT_CE_WEIGHT} \
      TRAINER.ANCHOROPT.GUMBEL_TEMP ${GUMBEL_TEMP} \
      TRAINER.ANCHOROPT.MAX_TEMPLATE_LENGTH 5 \
      TRAINER.ANCHOROPT.KD_TEMPERATURE ${KD_TEMPERATURE} \
      TRAINER.ANCHOROPT.KD_WEIGHT ${KD_WEIGHT} \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES all \
      OPTIM.MAX_EPOCH ${EPO}
done

