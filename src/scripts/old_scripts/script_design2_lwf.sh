#design 
python3 main_incremental.py --exp_name design2_lwf_lr00005 --datasets birds --num_tasks 4 --network disentangle --disentangle_design design2 --nepochs 200 --batch_size 64 --results_path data/experiments/4tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=0

python3 main_incremental.py --exp_name design2_lwf_lr0001 --datasets birds --num_tasks 4 --network disentangle --disentangle_design design2 --nepochs 200 --batch_size 64 --results_path data/experiments/4tasks_birds/ --lr 0.001 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=0


