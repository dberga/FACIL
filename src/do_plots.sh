python3 plot_incremental.py --noplot --results_path=data/experiments/1tasks_cifar100
python3 plot_incremental.py --noplot --results_path=data/experiments/1tasks_flowers
python3 plot_incremental.py --noplot --results_path=data/experiments/1tasks_birds
python3 plot_incremental.py --noplot --results_path=data/experiments/4tasks_flowers
python3 plot_incremental.py --noplot --results_path=data/experiments/10tasks_flowers
python3 plot_incremental.py --noplot --results_path=data/experiments/4tasks_birds
python3 plot_incremental.py --noplot --results_path=data/experiments/10tasks_birds
python3 plot_incremental.py --noplot --results_path=data/experiments/4tasks_cifar100
python3 plot_incremental.py --noplot --results_path=data/experiments/10tasks_cifar100
cp data/experiments/1tasks_*/*.csv .
cp data/experiments/4tasks_*/*.csv .
cp data/experiments/10tasks_*/*.csv .
