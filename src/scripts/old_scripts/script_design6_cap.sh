#design 5
python3 main_incremental.py --exp_name design6_lr00005_cap --datasets birds --num_tasks 4 --network disentangle --disentangle_design design6 --nepochs 200 --batch_size 32 --results_path data/experiments/4tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=2



