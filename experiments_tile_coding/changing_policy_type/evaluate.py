from experiments_tile_coding.evaluate import plot_all_experiments_training_stats

if __name__ == '__main__':
    exp_subdirs = [
        'eps-greedy',
        'boltz-10',
        'boltz-5',
        'boltz-1'
    ]
    exp_labels = [
        'Eps-greedy',
        'Boltz10',
        'Boltz5',
        'Boltz1'
    ]
    plot_all_experiments_training_stats('.', exp_subdirs, exp_labels)
