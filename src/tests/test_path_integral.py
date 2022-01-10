from tests import run_main

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --results_path ../results/" \
                       " --nepochs 3" \
                       " --num_workers 0" \
                       " --approach path_integral"


def test_pi_without_exemplars():
    run_main(FAST_LOCAL_TEST_ARGS)