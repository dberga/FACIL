import multiprocessing

import pytest

# easy debug:
multiprocessing.set_start_method('spawn', True)

from main_incremental import main

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --results_path ../results/" \
                       " --nepochs 2 --lr_factor 10 --momentum 0.9 --lr_min 1e-7" \
                       " --num_workers 0"


def test_finetuning_without_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    print('ARGS:', args_line)
    main(args_line.split(' '))


def test_finetuning_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars 200"
    print('ARGS:', args_line)
    main(args_line.split(' '))


def test_finetuning_with_exemplars_per_class_and_herding():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection herding"
    print('ARGS:', args_line)
    main(args_line.split(' '))


def test_finetuning_with_exemplars_per_class_and_entropy():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection entropy"
    print('ARGS:', args_line)
    main(args_line.split(' '))


def test_finetuning_with_exemplars_per_class_and_distance():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection distance"
    print('ARGS:', args_line)
    main(args_line.split(' '))


def test_wrong_args():
    with pytest.raises(SystemExit):  # error of providing both args
        args_line = FAST_LOCAL_TEST_ARGS
        args_line += " --approach finetune"
        args_line += " --num_exemplars_per_class 10"
        args_line += " --num_exemplars 200"
        print('ARGS:', args_line)
        main(args_line.split(' '))
