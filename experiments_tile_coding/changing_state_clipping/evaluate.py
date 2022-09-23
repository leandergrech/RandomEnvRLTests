from experiments_tile_coding.evaluate import plot_all_experiments_training_stats

if __name__ == '__main__':
    exp_subdirs = [
        'no-clipping',
        'clip-1.5',
        'clip-1.2',
        'clip-1.0'
    ]
    exp_labels = [
        'No clip',
        'Clip 1.5',
        'Clip 1.2',
        'Clip 1.0'
    ]
    plot_all_experiments_training_stats('.', exp_subdirs, exp_labels)
