#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running design: $1"
else
    echo "No design has been given."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

for APPROACH in finetune lwf joint
do
for DATASET in birds
do
for NUMTASKS in 4
do

if [ "$1" = "resnet18" ]; then
    python3.7 -u main_incremental.py --exp_name disentangle_val_resnet18 \
              --datasets $DATASET --num_tasks $NUMTASKS --network resnet18 --use_valid_only \
              --nepochs 200 --batch_size 64 --results_path /data/users/mmasana/Disentangle/ \
              --lr 0.001 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 \
              --approach $APPROACH --gpu $2
else
    python3.7 -u main_incremental.py --exp_name disentangle_val_${1} --disentangle_design $1 \
              --datasets $DATASET --num_tasks $NUMTASKS --network disentangle --use_valid_only \
              --nepochs 200 --batch_size 64 --results_path /data/users/mmasana/Disentangle/ \
              --lr 0.001 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 \
              --approach $APPROACH --gpu $2
fi

done
done
done
