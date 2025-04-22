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
import random
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nn_utils.dataset.base_dataset import BaseDataset
from nn_utils.policy.base_policy import BasePolicy

class BaseBehavioralCloning:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_dir = f"logs/{self.cfg.training_session_name}"
        self.ckpt_dir = f"ckpts/{self.cfg.training_session_name}"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        seed = self.cfg.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.learner = BasePolicy()

        self.dataset = BaseDataset(dataset_path=self.cfg.dataset_path,
                                   state_dim=self.cfg.state_dim,
                                   action_dim=self.cfg.action_dim)

        self.dataloader = DataLoader(self.dataset,
                                    batch_size=self.cfg.dataloader.batch_size,
                                    shuffle=self.cfg.dataloader.shuffle,
                                    num_workers=self.cfg.dataloader.num_workers)

        self.curr_epoch_idx = 0



    def training_one_batch(self, batch_idx, batch):
        return 0

    def evaluation(self):
        raise NotImplementedError()

    def run(self):

        with tqdm(range(self.cfg.num_training_epochs), desc="BC Training Epochs") as pbar:
            for curr_epoch_idx in pbar:

                self.curr_epoch_idx = curr_epoch_idx

                losses = []

                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Batch Progress")):
                    loss = self.training_one_batch(batch_idx, batch)
                    losses.append(loss)

                mean_loss = np.mean(losses)
                self.writer.add_scalar("Training/Epoch Loss", mean_loss, curr_epoch_idx)

                if self.cfg.use_eval and (curr_epoch_idx % self.cfg.eval_and_save_every_n_epochs == 0):
                    self.evaluation()
                    self.learner.save(path=f"{self.ckpt_dir}/{self.curr_epoch_idx}.ckpt")
