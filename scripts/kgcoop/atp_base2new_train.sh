#!/bin/bash

# custom config
DATA=/root/prompt_dataset
CFG=vit_b16_ep100_ctxv4
CTP=end
SHOTS=16
CSC=False
TRAINER=KgCoOp_ATP

DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101

for WEIGHT in 2.0 4.0
do
for NCTX in 2 4
do 
for EPO in 100
do
for SEED in 1 2 3 4 5
do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/CTX_${NCTX}_WT_${WEIGHT}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/KgCoOp/${CFG}.yaml \
                --output-dir ${DIR} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.ATPROMPT.USE_ATPROMPT True \
                TRAINER.ATPROMPT.N_ATT1 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT2 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT3 ${NCTX} \
                TRAINER.COOP.W ${WEIGHT} \
                OPTIM.MAX_EPOCH ${EPO}
done
done
done
done

