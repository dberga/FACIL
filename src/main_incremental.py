import argparse
import importlib
import os
import time
from functools import reduce

import numpy as np
import torch

import approach
import utils
from datasets.data_loader import get_loaders
from loggers.exp_logger import MultiLogger
from networks import tvmodels, allmodels, set_tvmodel_head_var
from datasets.dataset_config import dataset_config
from networks.extra_layers import get_new_head_architecture


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='Incremental Learning Framework')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--results_path', type=str, default='/data/experiments/LLL/', help='(default=%(default)s)')
    parser.add_argument('--exp_name', default=None, type=str, help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='(default=%(default)s)', nargs='*')
    parser.add_argument('--save_models', action='store_true', help='(default=%(default)s)')
    # data args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='(default=%(default)s)', nargs='+')
    parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--pin_memory', default=False, type=bool, required=False, help='(default=%(default)d)')
    parser.add_argument('--num_tasks', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--nc_first_task', default=None, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--use_valid_only', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--stop_at_task', default=0, type=int, required=False, help='(default=%(default)d)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='(default=%(default)s)')
    parser.add_argument('--not_remove_existing_head', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--heads_architecture', default='linear', type=str, help='(default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetune', type=str, choices=approach.__all__,
                        help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=2, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='(default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_min', default=1e-4, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_factor', default=3, type=float, required=False, help='(default=%(default)s)')
    parser.add_argument('--lr_patience', default=5, type=int, required=False, help='(default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False, help='(default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--weight_decay', default=0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--warmup_nepochs', default=0, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--warmup_lr_factor', default=1.0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--multi_softmax', type=bool, help='(default=%(default)s)')
    # disentangle args
    parser.add_argument('--disentangle_design', default='vanilla', type=str, help='(default=%(default)s)')
    parser.add_argument('--use_bn', default=1, type=int, required=False, help='(default=%(default)s)') #bool was not recognizing True/False
    parser.add_argument('--disentangle_factors', default=[1.0, 1.0], nargs='+', type=float, help='(default=%(default)f)')
    parser.add_argument('--smooth_color', default=0, type=int, required=False, help='(default=%(default)s)') #bool was not recognizing True/False
    args, extra_args = parser.parse_known_args(argv)
    utils.seed_everything(seed=args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'

    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)

    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    if args.save_models is None:
        args.save_models=False
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)

    # logger.log_args(args)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
    ####################################################################################################################

    # Args -- Continual Learning Approach
    from approach.learning_approach import Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__))
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Network
    from networks.network import LLL_Net

    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # When doing disentangle, design has to be set
        if args.network == 'disentangle':
            init_model = net(pretrained=False, design=args.disentangle_design, use_bn=bool(args.use_bn), factors=args.disentangle_factors, smooth_color=args.smooth_color)
        else:
            # WARNING: fixed to pretrained False
            init_model = net(pretrained=False)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    if args.stop_at_task == 0:
        max_task = len(taskcla)
    else:
        max_task = args.stop_at_task

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.not_remove_existing_head)
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_tranin_ds = trn_loader[0].dataset
    transform, class_indices = first_tranin_ds.transform, first_tranin_ds.class_indices
    appr_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, logger=logger, **appr_args.__dict__)
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)
    # Visualize network
    params_count=utils.print_model_report(net,True)
    logger.log_result(params_count,name="capacity")
    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((len(taskcla), len(taskcla)))
    acc_tag = np.zeros((len(taskcla), len(taskcla)))
    forg_taw = np.zeros((len(taskcla), len(taskcla)))
    forg_tag = np.zeros((len(taskcla), len(taskcla)))
    for t, (_, ncla) in enumerate(taskcla):
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Early stop tasks if flag
        if t >= max_task:
            continue

        # Add head for current task
        if args.heads_architecture != 'linear':
            net.model.avgpool=torch.nn.Sequential()
            #patch (out_size could be flattened or not)
            if net.out_size == (512*7*7):
                net.out_size=512
            elif net.out_size == (1024*7*7):
                net.out_size=1024
        taskhead=get_new_head_architecture(net.out_size, taskcla[t][1], args.heads_architecture)
        taskhead.out_features=taskcla[t][1]
        net.add_head(taskhead)
        net.to(device)

        #task model capacity
        params_count=utils.print_model_report(net,False)
        logger.log_result(params_count,name="".join(["capacity-",str(t)]))
        
        # Train
        appr.train(t, trn_loader[t], val_loader[t])
        print('-' * 108)

        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw")
        logger.log_result(acc_tag, name="acc_tag")
        logger.log_result(forg_taw, name="forg_taw")
        logger.log_result(forg_tag, name="forg_tag")
        logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw")
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag")
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla]], len(taskcla), axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw")
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag")

    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    ####################################################################################################################


if __name__ == '__main__':
    main()
