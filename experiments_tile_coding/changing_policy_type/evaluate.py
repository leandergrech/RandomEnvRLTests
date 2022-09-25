from experiments_tile_coding.evaluate import *

from random_env.envs import REDAClip

if __name__ == '__main__':
    exp_labels = [
        'Eps-greedy',
        'Boltz10',
        'Boltz5',
        'Boltz1'
    ]
    exp_subdirs = [
        'eps-greedy',
        'boltz-10',
        'boltz-5',
        'boltz-1'
    ]
    # plot_all_experiments_training_stats('1.0-clip', exp_subdirs, exp_labels)

    exp_pardir = 'no-clip/sarsa_092522_202747_0'
    # plot_experiment_training_stats(exp_pardir=exp_pardir, exp_subdirs=exp_subdirs, exp_labels=exp_labels)


    exp_dir = os.path.join(exp_pardir, 'boltz-1')
    plot_trims_during_training(exp_dir, REDAClip)
