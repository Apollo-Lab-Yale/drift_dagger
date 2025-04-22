if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import numpy as np
import random
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from nn_utils.dataset.base_dataset import BaseDataset

class BaseOfflineDataCollection:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg

        seed = self.cfg.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.dataset = BaseDataset(dataset_path=self.cfg.dataset_path,
                                   state_dim=self.cfg.state_dim,
                                   action_dim=self.cfg.action_dim)

        self.expert = None
        self.env = None
        self.obs = None
        self.done = False


    def reset(self):
        raise NotImplementedError()

    def collect_current_step_data(self):
        raise NotImplementedError()

    def run(self):

        with tqdm(range(self.cfg.num_of_rollouts), desc="Data Collection Rollouts") as pbar:
            for rollout_idx in pbar:

                self.reset()
                step_idx = 0

                self.dataset.create_new_rollout_dataset()

                while not self.done and step_idx < self.cfg.max_steps_per_rollout:
                    self.collect_current_step_data()
                    step_idx += 1

                print("step_idx: ", step_idx)
                self.dataset.stop_recording_curr_rollout_dataset_and_add_it_to_this_object()