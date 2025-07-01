#!/bin/bash

DATA=/root/prompt_dataset

CFG=vit_b16  # config file
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

TRAINER=AttributeCompute
CTP=end

NCTX=4
SEED=1

for DATASET in caltech101
do
if [ ${DATASET} = "imagenet" ]; then
        ATT1_TEXT=color
        ATT2_TEXT=size
        ATT3_TEXT=shape
        ATT4_TEXT=habitat
        ATT5_TEXT=behavior
        echo 'imagenet'
elif [ ${DATASET} = "caltech101" ]; then
        ATT1_TEXT=shape
        ATT2_TEXT=color
        ATT3_TEXT=material
        ATT4_TEXT=function
        ATT5_TEXT=size
        echo 'caltech'
elif [ $DATASET = "oxford_pets" ]; then
        ATT1_TEXT=loyalty
        ATT2_TEXT=affection
        ATT3_TEXT=playfulness
        ATT4_TEXT=energy
        ATT5_TEXT=intelligence
        echo 'pets'
elif [ $DATASET = "stanford_cars" ]; then
        ATT1_TEXT=design
        ATT2_TEXT=engine
        ATT3_TEXT=performance
        ATT4_TEXT=luxury
        ATT5_TEXT=color
        echo 'cars'
elif [ $DATASET = 'oxford_flowers' ]; then
        ATT1_TEXT=color
        ATT2_TEXT=flower
        ATT3_TEXT=habitat
        ATT4_TEXT=growth
        ATT5_TEXT=season
        echo 'flowers'
elif [ $DATASET = 'food101' ]; then
        ATT1_TEXT=flavor
        ATT2_TEXT=texture
        ATT3_TEXT=origin
        ATT4_TEXT=ingredients
        ATT5_TEXT=preparation
        echo 'food'
elif [ $DATASET = 'fgvc_aircraft' ]; then
        ATT1_TEXT=design
        ATT2_TEXT=capacity
        ATT3_TEXT=range
        ATT4_TEXT=engines
        ATT5_TEXT=liveries
        echo 'fgvc'
elif [ $DATASET = 'sun397' ]; then
        ATT1_TEXT=architecture
        ATT2_TEXT=environment
        ATT3_TEXT=structure
        ATT4_TEXT=design
        ATT5_TEXT=function
        echo 'sun'
elif [ $DATASET = 'dtd' ]; then
        ATT1_TEXT=pattern
        ATT2_TEXT=texture
        ATT3_TEXT=color
        ATT4_TEXT=design
        ATT5_TEXT=structure
        echo 'dtd'
elif [ $DATASET = 'eurosat' ]; then
        ATT1_TEXT=habitat
        ATT2_TEXT=foliage
        ATT3_TEXT=infrastructure
        ATT4_TEXT=terrain
        ATT5_TEXT=watercourse
        echo 'eurosat'
elif [ $DATASET = 'ucf101' ]; then
        ATT1_TEXT=precision
        ATT2_TEXT=coordination
        ATT3_TEXT=technique
        ATT4_TEXT=strength
        ATT5_TEXT=control
        echo 'ucf'
else
        echo 'no value'
fi
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/search_attribute

        CUDA_VISIBLE_DEVICES=0 python train_select_attribute.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/CoOp/${CFG}.yaml \
                --output-dir ${DIR} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.COOP.N_ATT1 ${NCTX} \
                TRAINER.COOP.N_ATT2 ${NCTX} \
                TRAINER.COOP.N_ATT3 ${NCTX} \
                TRAINER.COOP.N_ATT4 ${NCTX} \
                TRAINER.COOP.N_ATT5 ${NCTX} \
                TRAINER.COOP.ATT1_TEXT ${ATT1_TEXT} \
                TRAINER.COOP.ATT2_TEXT ${ATT2_TEXT} \
                TRAINER.COOP.ATT3_TEXT ${ATT3_TEXT} \
                TRAINER.COOP.ATT4_TEXT ${ATT4_TEXT} \
                TRAINER.COOP.ATT5_TEXT ${ATT5_TEXT} \
                OPTIM.MAX_EPOCH 40
done

