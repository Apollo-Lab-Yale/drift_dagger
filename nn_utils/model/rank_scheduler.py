import numpy as np

class RankScheduler:
    def __init__(self,
                r_max,
                r_min,
                num_training_steps,
                decay_type='linear'):
        self.r_max = r_max
        # self.r_min = int(r_max * 0.25)
        self.r_min = r_min

        print("r_max: ", r_max)
        print("r_min: ", r_min)
        self.num_training_steps = num_training_steps
        self.decay_type = decay_type
        self.curr_step = 0

        self.t_mid = self.num_training_steps / 2

    def step_increment(self):
        if self.curr_step < self.num_training_steps:
            self.curr_step += 1

    def get_rank(self, **kwargs):
        if self.decay_type == 'linear':
            return self.linear_decay()
        elif self.decay_type == 'cosine':
            return self.cosine_decay()
        elif self.decay_type == 'sigmoid':
            return self.sigmoid_decay(kwargs['steepness'])
        elif self.decay_type == 'exponential':
            return self.exponential_decay(kwargs['steepness'])
        elif self.decay_type == 'log':
            return self.log_decay(kwargs['base'])
        else:
            raise ValueError('Invalid decay type')

    def linear_decay(self):
        r = int(self.r_max + (self.r_min - self.r_max) * (self.curr_step / self.num_training_steps))
        return r

    def cosine_decay(self):
        r = int(self.r_min + 0.5 * (self.r_max - self.r_min) * (1 + np.cos(np.pi * self.curr_step / self.num_training_steps)))
        return r

    def sigmoid_decay(self, steepness=0.1):
        r = self.r_max - (self.r_max - self.r_min) / (1 + np.exp(-steepness * (self.curr_step - self.t_mid)))
        return int(r)

    def exponential_decay(self, steepness=0.1):
        r = self.r_min + (self.r_max - self.r_min) * (np.exp(-steepness * self.curr_step))
        return int(r)

    def log_decay(self, base=2):
        r = self.r_min + (self.r_max - self.r_min) / (1 + np.log(self.curr_step + 1) / np.log(base))
        return int(r)