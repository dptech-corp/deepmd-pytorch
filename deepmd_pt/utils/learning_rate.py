import numpy as np
import math


class LearningRateExp(object):

    def __init__(self, start_lr, stop_lr, decay_steps, stop_steps, **kwargs):
        """Construct an exponential-decayed learning rate.

        Args:
        - start_lr: Initial learning rate.
        - stop_lr: Learning rate at the last step.
        - decay_steps: Decay learning rate every N steps.
        - stop_steps: When is the last step.
        """
        self.start_lr = start_lr
        default_ds = 100 if stop_steps // 10 > 100 else stop_steps // 100 + 1
        self.decay_steps = decay_steps
        if self.decay_steps >= stop_steps:
            self.decay_steps = default_ds
        self.decay_rate = np.exp(np.log(stop_lr / self.start_lr) / (stop_steps / self.decay_steps))
        if 'decay_rate' in kwargs:
            self.decay_rate = kwargs['decay_rate']
        if 'min_lr' in kwargs:
            self.min_lr = kwargs['min_lr']
        else:
            self.min_lr = 3e-10

    def value(self, step):
        """Get the learning rate at the given step."""
        step_lr = self.start_lr * np.power(self.decay_rate, step // self.decay_steps)
        if step_lr < self.min_lr:
            step_lr = self.min_lr
        return step_lr


class LearningRatePolynomial(object):

    def __init__(self, start_lr, stop_lr, stop_steps, **kwargs):
        """Construct an exponential-decayed learning rate.

        Args:
        - start_lr: Initial learning rate.
        - stop_lr: Learning rate at the last step.
        - stop_steps: When is the last step.
        - decay_steps: Given a provided start_lr, to reach stop_lr in the given decay_steps.
        - power: The power of the polynomial. Defaults to 1(Linear)
        - cycle: whether it should cycle beyond decay_steps
        """
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.decay_steps = kwargs.get('decay_steps', stop_steps)
        self.power = kwargs.get('power', 1)
        self.cycle = kwargs.get('cycle', False)

    def value(self, step):
        """Get the learning rate at the given step."""
        if not self.cycle:
            step = min(step, self.decay_steps)
            step_lr = self.stop_lr + (self.start_lr - self.stop_lr) * pow((1 - step / self.decay_steps), self.power)
        else:
            decay_steps = self.decay_steps * math.ceil(step / self.decay_steps)
            if(step == 0):
                decay_steps = self.decay_steps
            step_lr = self.stop_lr + (self.start_lr - self.stop_lr) * pow((1 - step / decay_steps), self.power)

        return step_lr