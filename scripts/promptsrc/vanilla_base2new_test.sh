#!/bin/bash

# custom config
DATA=/root/prompt_dataset
TRAINER=PromptSRC

CFG=vit_b16_c2_ep20_batch4_4+4ctx
SHOTS=16

DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101

for NCTX in 4
do
for EPO in 5 10
do
for SEED in 1 2 3 4 5 
do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/PromptSRC/${CFG}.yaml \
                --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
                --model-dir ${DIR} \
                --load-epoch ${EPO} \
                --eval-only \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES new \
                TRAINER.PROMPTSRC.N_CTX_TEXT ${NCTX} \
                TRAINER.PROMPTSRC.N_CTX_VISION ${NCTX} \
                OPTIM.MAX_EPOCH ${EPO}
done
done
done
