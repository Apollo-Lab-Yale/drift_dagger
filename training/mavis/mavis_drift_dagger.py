if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
os.environ['MUJOCO_GL'] = 'egl'
import cv2
import hydra
import collections
import numpy as np
from tqdm.auto import tqdm
import torch
from diffusers.optimization import get_scheduler
from nn_utils.policy.visuomotor_policy_with_lora import (VisuomotorUNetDiffusionPolicyWithLoRA,
                                                         VisuomotorTransformerDiffusionPolicyWithLoRA)
from training.mavis.mavis_hg_dagger import MAVISHumanGatedDataAggregation
from mavis_mujoco_gym.utils.create_env import create_env

class MAVISLowRankExpertGatedDataAggregation(MAVISHumanGatedDataAggregation):
    def __init__(self, cfg):
        super().__init__(cfg)

    def bootstrapping(self):
        super().bootstrapping()






    def training_one_epoch(self):
        #if self.finished_bootstrapping:

                # Update the optimizer with parameter groups
                #self.optimizer = torch.optim.Adam([
                #    {'params': lora_params, 'lr': self.cfg.optimizer.learning_rate},
                #    {'params': main_params, 'lr': 0.0}
                #], weight_decay=self.cfg.optimizer.weight_decay)

                # Update the scheduler
                #self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler,
                #                                  optimizer=self.optimizer,
                #                                  num_warmup_steps=self.cfg.lr_warmup_steps,
                #                                  num_training_steps=(
                #                                  (self.cfg.num_iter * self.cfg.num_epoch_per_iter) *
                #                                  (self.cfg.num_iter * self.cfg.num_rollout_per_iter *
                #                                   self.cfg.max_num_steps_per_rollout / self.cfg.dataloader.batch_size)
                #                          ))

        mean_loss = super().training_one_epoch()

        #if self.finished_bootstrapping:
        #    if self.learner.lora_injected and self.cfg.network.merge_lora_after_each_epoch:
        #        self.learner.merge_lora_weights(self.cfg.network.apply_lora_to_visual_encoder)

        #        self.update_optimizer_parameters()
        #        self.update_scheduler()

                # After merging, update the optimizer to include all parameters
                #self.optimizer = torch.optim.Adam(self.learner.parameters(),
                #                                  lr=self.cfg.optimizer.learning_rate,
                #                                  weight_decay=self.cfg.optimizer.weight_decay)

                # Update the scheduler
                #self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler,
                #                                  optimizer=self.optimizer,
                #                                  num_warmup_steps=self.cfg.lr_warmup_steps,
                #                                  num_training_steps=(
                #                                  (self.cfg.num_iter * self.cfg.num_epoch_per_iter) *
                #                                  (self.cfg.num_iter * self.cfg.num_rollout_per_iter *
                #                                   self.cfg.max_num_steps_per_rollout / self.cfg.dataloader.batch_size)
                #                          ))

        return mean_loss



@hydra.main(
    version_base=None,
    config_path='../../config',
    config_name="mavis_pick_and_place_drift_dagger_dp_unet")
def main(cfg):
    mavis_drift_dagger = MAVISLowRankExpertGatedDataAggregation(cfg)
    mavis_drift_dagger.run()

if __name__ == "__main__":
    main()
