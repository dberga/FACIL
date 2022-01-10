#birds
python3 main_incremental.py --exp_name design7_finetune_lr005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design7 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=$1

python3 main_incremental.py --exp_name design6_finetune_lr005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=$1

python3 main_incremental.py --exp_name design2_finetune_lr0005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design2 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=$1

python3 main_incremental.py --exp_name shapeonly_finetune_lr0005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design shapeonly --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=$1







