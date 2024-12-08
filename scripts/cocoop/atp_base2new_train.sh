#!/bin/bash

# custom configs
DATA=/root/prompt_dataset
TRAINER=CoCoOp_ATP

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16
EPO=10

DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101

for NCTX in 2 4 
do
for SEED in 1 2 3 4 5
do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/CoCoOp/${CFG}.yaml \
                --output-dir ${DIR} \
                TRAINER.COCOOP.N_CTX ${NCTX} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.ATPROMPT.USE_ATPROMPT True \
                TRAINER.ATPROMPT.N_ATT1 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT2 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT3 ${NCTX} \
                OPTIM.MAX_EPOCH ${EPO}
done
done
