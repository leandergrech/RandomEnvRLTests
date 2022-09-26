from experiments_tile_coding.evaluate import *

from random_env.envs import REDAClip

if __name__ == '__main__':
    exp_labels = [
        'Eps-greedy',
        # 'Boltz10',
        # 'Boltz5',
        'Boltz1'
    ]
    exp_subdirs = [
        'eps-greedy',
        # 'boltz-10',
        # 'boltz-5',
        'boltz-1'
    ]
    plot_all_experiments_training_stats('.', exp_subdirs, exp_labels, exp_filter='1644')

    # exp_pardir = 'sarsa_092622_161435_0'
    # sub_exp = 'boltz-1'
    # plot_experiment_training_stats(exp_pardir, exp_subdirs, exp_labels)

    # exp_dir = os.path.join(exp_pardir, sub_exp)
    # exp_dir = os.path.join(exp_pardir, 'eps-greedy')
    # plot_trims_during_training(exp_dir, REDAClip)#, save_dir=os.path.join('log_trims', sub_exp))
