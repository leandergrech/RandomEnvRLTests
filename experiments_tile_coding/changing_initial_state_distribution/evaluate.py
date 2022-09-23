from experiments_tile_coding.evaluate import plot_all_experiments_training_stats

if __name__ == '__main__':
    exp_subdirs = [
        'circular-reset',
        'uniform-reset']
    exp_labels = [
        'Circular',
        'Uniform',
    ]
    plot_all_experiments_training_stats('.', exp_subdirs, exp_labels)
