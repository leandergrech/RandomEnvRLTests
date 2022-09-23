from experiments_tile_coding.evaluate import plot_all_experiments_training_stats

if __name__ == '__main__':
    exp_subdirs = [
        'quadratic-objective',
        'quadratic-objective-x5',
        'quadratic-objective-x10',
        'rms-objective'
    ]
    exp_labels = [
        'QO',
        'QOx5',
        'QOx10',
        'RO'
    ]
    plot_all_experiments_training_stats('.', exp_subdirs, exp_labels)
