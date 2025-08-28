#!/bin/bash

# custom config
DATA=/root/prompt_dataset
CFG=vit_b16_ep10_bs4_lr35
SHOTS=16

TRAINER=DePT_ATP
EPO=10
NCTX=2
WEIGHT=0.7
DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenet

for SEED in 1 2 3 4 5
do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/DePT/${CFG}.yaml \
                --output-dir ${DIR} \
                DATASET.NUM_SHOTS 16 \
                DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.LINEAR_PROBE.WEIGHT ${WEIGHT} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.ATPROMPT.USE_ATPROMPT True \
                TRAINER.ATPROMPT.N_ATT1 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT2 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT3 ${NCTX}
done

