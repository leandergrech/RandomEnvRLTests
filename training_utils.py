import yaml
def init_label(label):
    if label:
        return f'{label}_'
    else:
        return ''


class CustomFunc:
    def __init__(self):
        self.pow = 0.75

    def __call__(self, ep_idx):
        return 1/((ep_idx + 1) ** self.pow)

    def __repr__(self):
        return f'1/((ep_idx + 1) ** {self.pow}'


class Constant:
    def __init__(self, val, label=''):
        self.val = val
        self.label = init_label(label)

    def __call__(self, *args, **kwargs):
        return self.val

    def __repr__(self):
        return f'{self.label}{self.val}'


class ExponentialDecay:
    def __init__(self, init, halflife, label=None):
        self.init = init
        self.halflife = halflife
        self.label = init_label(label)

    def __call__(self, t):
        return self.init / (1. + (t / self.halflife))

    def __repr__(self):
        return f'{self.label}ExponentialDecay_init-{self.init}_halflife-{self.halflife}'


class LinearDecay:
    def __init__(self, init, final, decay_steps, label=None):
        self.init = init
        self.final = final
        self.decay_steps = decay_steps
        self.label = init_label(label)

    def __call__(self, t):
        return self.final + (self.init - self.final) * max(0, 1 - t/self.decay_steps)

    def __repr__(self):
        return f'{self.label}LinearDecay_init-{self.init}_final-{self.final}_decaysteps-{self.decay_steps}'


class StepDecay:
    def __init__(self, init, decay_rate, decay_every, label=None):
        self.init = init
        self.decay_rate = decay_rate
        self.decay_every = decay_every
        self.label = init_label(label)

    def __call__(self, t):
        return self.init * self.decay_rate**(t//self.decay_every)

    def __repr__(self):
        return f'{self.label}StepDecay_init-{self.init}_decayrate-{self.decay_rate}_decayevery-{self.decay_every}'

def customFunc_representer(dumper, fun):
    return dumper.represent_mapping("!CustomFunc", {
        "func": repr(fun)
    })

def constant_representer(dumper, fun):
    return dumper.represent_mapping("!Constant", {
        "val": fun.val
    })
def exponentialDecay_representer(dumper: yaml.SafeDumper, fun: ExponentialDecay) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!ExponentialDecay", {
        "init": fun.init,
        "halflife": fun.halflife,
        "label": fun.label
    })


def linearDecay_representer(dumper: yaml.SafeDumper, fun: LinearDecay) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!LinearDecay", {
        "init": fun.init,
        "final": fun.final,
        "decay_steps": fun.decay_steps,
        "label": fun.label
    })


def stepDecay_representer(dumper: yaml.SafeDumper, fun: StepDecay) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!StepDecay", {
        "init": fun.init,
        "decay_rate": fun.decay_rate,
        "decay_every": fun.decay_every,
        "label": fun.label
    })

def get_training_utils_yaml_dumper():
    safe_dumper = yaml.SafeDumper
    safe_dumper.add_representer(CustomFunc, customFunc_representer)
    safe_dumper.add_representer(Constant, constant_representer)
    safe_dumper.add_representer(ExponentialDecay, exponentialDecay_representer)
    safe_dumper.add_representer(LinearDecay, linearDecay_representer)
    safe_dumper.add_representer(StepDecay, stepDecay_representer)
    return safe_dumper


if __name__ == '__main__':
    # fun = Constant(1., 'LR')
    # fun = StepDecay(1., 0.5, 2)
    # fun = LinearDecay(2., 1., 5)
    fun = ExponentialDecay(1e-1, 100000, 'LR')

    vals = [fun(x) for x in range(5000)]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(vals)
    plt.show()

