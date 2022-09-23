from experiments_tile_coding.evaluate import plot_all_experiments_training_stats

if __name__ == '__main__':
    exp_subdirs = [
        'REDAClip_1.0clip_2obsx2act',
        'REDAClip_1.0clip_3obsx2act',
        'REDAClip_1.0clip_4obsx2act',
        'REDAClip_1.0clip_5obsx2act',
    ]
    exp_labels = [
        '2obsx2act',
        '3obsx2act',
        '4obsx2act',
        '5obsx2act',
    ]
    plot_all_experiments_training_stats('.', exp_subdirs, exp_labels)
