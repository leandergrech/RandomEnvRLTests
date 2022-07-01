def lr(init_lr, halflife):
    calls = 0.
    while True:
        if halflife <= 0:
            yield init_lr
        else:
            yield init_lr/(1. + (1./halflife)*calls)
        calls += 1.