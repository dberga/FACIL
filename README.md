<div align="center">

# Disentanglement of Color and Shape for CL using the Framework for Analysis of Class-Incremental Learning (FACIL)
[![](https://i.ytimg.com/vi/NZLIm0uPKEY/hqdefault.jpg)](https://www.youtube.com/watch?v=NZLIm0uPKEY)
---

<p align="center">
  <a href="#disentanglement-implementation">Disentanglement CL-ICML</a> •
  <a href="#example-execution-scripts">What is FACIL</a> •
  <a href="#what-is-facil">What is FACIL</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#simple-installation-with-conda">Simple installation with conda</a> •
  <a href="#how-to-use-facil">How To Use FACIL</a> •
  <a href="src/approach#approaches-1">Approaches</a> •
  <a href="src/datasets#datasets">Datasets</a> •
  <a href="src/networks#networks">Networks</a> •
  <a href="#license">License</a> •
  <a href="#cite">Cite</a>
</p>
</div>

---
## Disentanglement Implementation
CL-ICML paper:
_**Disentanglement of Color and Shape Representations for Continual Learning**_  
*David Berga, Marc Masana, Joost van de Weijer*  
([arxiv](https://arxiv.org/abs/2007.06356))
([CL-ICML 2020](https://sites.google.com/view/cl-icml/accepted-papers))

## Example execution scripts

See all script commands in scripts/jobs_10tasks.sh
Run different designs with lr 0.05, 0.005, 0.0005 with the following commands:

ResNet18-DS
```
python3 main_incremental.py --exp_name resnet18ds_finetune_lr005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design7 --heads_architecture design7 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18ds_finetune_lr0005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design7 --heads_architecture design7 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18ds_finetune_lr00005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design design7 --heads_architecture design7 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
```
ResNet18
```
python3 main_incremental.py --exp_name resnet18_finetune_lr005_bs32 --datasets birds --num_tasks 10 --network resnet18 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18_finetune_lr0005_bs32 --datasets birds --num_tasks 10 --network resnet18 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18_finetune_lr00005_bs32 --datasets birds --num_tasks 10 --network resnet18 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
```
ResNet18-H
```
python3 main_incremental.py --exp_name resnet18_finetune_lr005_bs32 --datasets birds --num_tasks 10 --network resnet18 --heads_architecture design7 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18_finetune_lr0005_bs32 --datasets birds --num_tasks 10 --network resnet18 --heads_architecture design7  --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18_finetune_lr00005_bs32 --datasets birds --num_tasks 10 --network resnet18 --heads_architecture design7 --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
```
ResNet18-Shape
```
python3 main_incremental.py --exp_name resnet18shape_finetune_lr005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design shapeonly --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18shape_finetune_lr0005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design shapeonly --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18shape_finetune_lr00005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design shapeonly --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
```
ResNet18-Color
```
python3 main_incremental.py --exp_name resnet18color_finetune_lr005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design coloronly --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.05 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18color_finetune_lr0005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design coloronly --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
python3 main_incremental.py --exp_name resnet18color_finetune_lr00005_bs32 --datasets birds --num_tasks 10 --network disentangle --disentangle_design coloronly --nepochs 200 --batch_size 32 --results_path data/experiments/10tasks_birds/ --lr 0.0005 --lr_factor 3 --lr_patience 15 --lr_min 1e-6 --momentum 0.9 --weight_decay 0.0002 --approach finetune --gpu=0 ;
```

## What is FACIL
FACIL started as code for the paper:  
_**Class-incremental learning: survey and performance evaluation**_  
*Marc Masana, Xialei Liu, Bartlomiej Twardowski, Mikel Menta, Andrew D. Bagdanov, Joost van de Weijer*  
([arxiv](https://arxiv.org/abs/2010.15277))

It allows to reproduce the results in the paper as well as provide a (hopefully!) helpful framework to develop new
methods for incremental learning and analyse existing ones. Our idea is to expand the available approaches
and tools with the help of the community. To help FACIL grow, don't forget to star this github repository and
share it to friends and coworkers!

## Key Features
We provide a framework based on class-incremental learning. However, task-incremental learning is also fully
supported. Experiments by default provide results on both task-aware and task-agnostic evaluation. Furthermore, if an
experiment runs with one task on one dataset, results would be equivalent to 'common' supervised learning.

| Setting | task-ID at train time | task-ID at test time | # of tasks |
| -----   | ------------------------- | ------------------------ | ------------ |
| [class-incremental learning](https://arxiv.org/pdf/2010.15277.pdf) | yes | no | ≥1 |
| [task-incremental learning](https://ieeexplore.ieee.org/abstract/document/9349197) | yes | yes | ≥1 |
| non-incremental supervised learning | yes | yes | 1 |

Current available approaches include:
<div align="center">
<p align="center"><b>
  Finetuning • Freezing • Joint

  LwF • iCaRL • EWC • PathInt • MAS • RWalk • EEIL • LwM • DMC • BiC • LUCIR • IL2M
</b></p>
</div>


## Simple installation with conda

Run this with miniconda
```
conda create -n deepenv python=3.5
conda install -n deepenv pytorch torchvision python-graphviz matplotlib numpy scipy intel-openmp mkl cudatoolkit=9.2 -c pytorch
conda activate deepenv

## How To Use FACIL
Clone this github repository:
```
git clone https://github.com/mmasana/FACIL.git
cd FACIL
```

<details>
  <summary>Optionally, create an environment to run the code (click to expand).</summary>

  ### Using a requirements file
  The library requirements of the code are detailed in [requirements.txt](requirements.txt). You can install them
  using pip with:
  ```
  python3 -m pip install -r requirements.txt
  ```

  ### Using a conda environment
  Development environment based on Conda distribution. All dependencies are in `environment.yml` file.

  #### Create env
  To create a new environment check out the repository and type: 
  ```
  conda env create --file environment.yml --name FACIL
  ```
  *Notice:* set the appropriate version of your CUDA driver for `cudatoolkit` in `environment.yml`.

  #### Environment activation/deactivation
  ```
  conda activate FACIL
  conda deactivate
  ```

</details>

To run the basic code:
```
python3 -u src/main_incremental.py
```
More options are explained in the [`src`](./src), including GridSearch usage. Also, more specific options on approaches,
loggers, datasets and networks.


## License
Please check the MIT license that is listed in this repository.

## Cite
If you want to cite the framework / code feel free to use this preprint citations:

```bibtex
@article{masana2020class,
  title={Class-incremental learning: survey and performance evaluation},
  author={Masana, Marc and Liu, Xialei and Twardowski, Bartlomiej and Menta, Mikel and Bagdanov, Andrew D and van de Weijer, Joost},
  journal={arXiv preprint arXiv:2010.15277},
  year={2020}
}
@article{berga2020disentanglement,
  title={Class-incremental learning: survey and performance evaluation},
  author={Berga, David and Masana, Marc and van de Weijer, Joost},
  journal={arXiv preprint arXiv:2007.06356},
  year={2020}
}
```
*Note: this implementation of Disentanglement is using an outdated version of FACIL, please check latest updates in the original [repo](https://github.com/mmasana/FACIL)


---

The basis of FACIL is made possible thanks to [Marc Masana](https://github.com/mmasana),
[Xialei Liu](https://github.com/xialeiliu), [Bartlomiej Twardowski](https://github.com/btwardow)
and [Mikel Menta](https://github.com/mkmenta). Code structure is inspired by [HAT](https://github.com/joansj/hat.). Feel free to contribute or propose new features by opening an issue!
