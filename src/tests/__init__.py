import os

from main_incremental import main


def run_main(args_line):
    if os.getcwd().endswith('tests'):
        os.chdir('../..')
    elif os.getcwd().endswith('src'):
        os.chdir('..')
    print('CWD:', os.getcwd())
    print('ARGS:', args_line)
    main(args_line.split(' '))
