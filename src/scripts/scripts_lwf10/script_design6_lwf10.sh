#birds
python3 main_incremental.py --exp_name design6_lwf_lr00005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

python3 main_incremental.py --exp_name design6_lwf_lr0005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

python3 main_incremental.py --exp_name design6_lwf_lr005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

#flowers

python3 main_incremental.py --exp_name design6_lwf_lr00005_bs32 --datasets flowers --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_flowers/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

python3 main_incremental.py --exp_name design6_lwf_lr0005_bs32 --datasets flowers --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_flowers/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

python3 main_incremental.py --exp_name design6_lwf_lr005_bs32 --datasets flowers --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_flowers/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

#cifar100
#python3 main_incremental.py --exp_name design6_lwf_lr00005_bs32 --datasets cifar100 --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_cifar100/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

#python3 main_incremental.py --exp_name design6_lwf_lr0005_bs32 --datasets cifar100 --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_cifar100/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1

#python3 main_incremental.py --exp_name design6_lwf_lr005_bs32 --datasets cifar100 --num_tasks 10 --network disentangle --disentangle_design design6 --heads_architecture design6 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_cifar100/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach lwf --lamb 1 --T 2 --gpu=$1


