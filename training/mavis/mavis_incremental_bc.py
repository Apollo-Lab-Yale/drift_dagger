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
import torch
import random
import collections
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler
from training.base.base_online_training import BaseOnlineTraining
from nn_utils.policy.visuomotor_policy import (VisuomotorUNetDiffusionPolicy,
                                               VisuomotorTransformerDiffusionPolicy)
from nn_utils.dataset.visuomotor_dataset import RecedingHorizonVisuomotorDataset
from nn_utils.tensorboard_utils import save_video_from_tensor
from expert_policy.mavis_legacy_policy import MAVISLegacyExpertPolicy
from mavis_mujoco_gym.utils.create_env import create_env

class MAVISIncrementalBehavioralCloning(BaseOnlineTraining):
    def __init__(self, cfg):
        super().__init__(cfg)



        self.expert = MAVISLegacyExpertPolicy(ckpt_file_dir=self.cfg.expert_policy_path,
                                              device=self.expert_device)



        self.optimizer = torch.optim.Adam(params=self.learner.parameters(),
                                          lr=self.cfg.optimizer.learning_rate,
                                          weight_decay=self.cfg.optimizer.weight_decay)

        self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler,
                                          optimizer=self.optimizer,
                                          num_warmup_steps=self.cfg.lr_warmup_steps,
                                          num_training_steps=(
                                              (self.cfg.num_iter * self.cfg.num_epoch_per_iter) *
                                              (self.cfg.num_iter * self.cfg.num_rollout_per_iter *
                                               self.cfg.max_num_steps_per_rollout / self.cfg.dataloader.batch_size)
                                            )
                                          )

        OmegaConf.save(config=self.cfg, f=os.path.join(self.ckpt_dir, "config.yaml"))

    def training_one_batch(self, batch_idx, batch):
        images, states, actions = batch
        images = images.float().to(self.learner_device)
        states = states.float().to(self.learner_device)
        actions = actions.float().to(self.learner_device)

        self.optimizer.zero_grad()
        loss = self.learner.compute_loss((images, states, actions))

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def collect_one_rollout(self, interactive=False):
        self.dataset.create_new_rollout_dataset()

        data_collection_env = create_env(env_id=self.cfg.env_id,
                                         env_configs=self.cfg.env_configs)
        obs, info = data_collection_env.reset()

        obs['rgb'] = cv2.resize(obs['rgb'], (self.cfg.downsampled_img_width, self.cfg.downsampled_img_height))
        obs_deque = collections.deque([obs] * self.cfg.network.obs_horizon, maxlen=self.cfg.network.obs_horizon)

        demo_step_idx = 0

        done = False
        terminated = False
        truncated = False

        with tqdm(total=self.cfg.max_num_steps_per_rollout, desc="Number of Demonstration Steps") as pbar:
            while demo_step_idx < self.cfg.max_num_steps_per_rollout and not done:
                images = np.stack([x['rgb'] for x in obs_deque])
                agent_poses = np.stack([x['state'] for x in obs_deque])

                expert_action = self.expert.predict_action((images, agent_poses))

                for i in range(len(expert_action)):
                    curr_expert_action = expert_action[i]

                    self.dataset.append_to_curr_rollout_dataset(rgb_img=obs['rgb'],
                                                                state=obs['state'],
                                                                action=curr_expert_action,
                                                                is_expert_label=True)
                    self.curr_expert_label_num += 1

                    obs, reward, terminated, truncated, info = data_collection_env.step(curr_expert_action)

                    obs['rgb'] = cv2.resize(obs["rgb"], (self.cfg.downsampled_img_width, self.cfg.downsampled_img_height))

                    done = terminated or truncated
                    obs_deque.append(obs)
                    demo_step_idx += 1
                    pbar.update(1)

                    if demo_step_idx > self.cfg.max_num_steps_per_rollout or truncated:
                        print("Current demonstration rollout exceeds maximum number of steps.")
                        self.dataset.remove_curr_rollout_dataset()
                        data_collection_env.close()
                        del data_collection_env
                        return False
                    if terminated:
                        self.dataset.stop_recording_curr_rollout_dataset_and_add_it_to_this_object()
                        data_collection_env.close()
                        del data_collection_env
                        return True

    def evaluation(self):
        eval_rollout_lengths = []
        is_eval_rollout_success = []
        log_imgs = None
        with tqdm(total=self.cfg.num_eval_rollouts, desc="Number of Evaluation Rollouts") as pbar:
            for eval_rollout_idx in range(self.cfg.num_eval_rollouts):
                eval_env = create_env(env_id=self.cfg.env_id,
                                      env_configs=self.cfg.env_configs)
                obs, info = eval_env.reset()

                obs['rgb'] = cv2.resize(obs['rgb'], (self.cfg.downsampled_img_width, self.cfg.downsampled_img_height))
                obs_deque = collections.deque([obs] * self.cfg.network.obs_horizon, maxlen=self.cfg.network.obs_horizon)

                eval_step_idx = 0

                done = False
                terminated = False
                truncated = False

                if eval_rollout_idx == self.cfg.num_eval_rollouts - 1:
                    log_imgs = [eval_env.render()]

                with tqdm(total=self.cfg.max_num_steps_per_rollout, desc="Number of Evaluation Steps") as pbar:
                    while eval_step_idx < self.cfg.max_num_steps_per_rollout and not done:
                        images = np.stack([x['rgb'] for x in obs_deque])
                        agent_poses = np.stack([x['state'] for x in obs_deque])

                        learner_action = self.learner.predict_action((images, agent_poses))

                        for i in range(len(learner_action)):
                            obs, reward, terminated, truncated, info = eval_env.step(learner_action[i])
                            obs['rgb'] = cv2.resize(obs["rgb"],(self.cfg.downsampled_img_width, self.cfg.downsampled_img_height))

                            if eval_rollout_idx == self.cfg.num_eval_rollouts - 1:
                                log_imgs.append(eval_env.render())

                            done = terminated or truncated
                            obs_deque.append(obs)
                            eval_step_idx += 1
                            pbar.update(1)

                            if eval_step_idx> self.cfg.max_num_steps_per_rollout or truncated:
                                is_eval_rollout_success.append(False)
                                done = True
                            if terminated:
                                is_eval_rollout_success.append(True)
                            if done:
                                break
                eval_rollout_lengths.append(eval_step_idx)
                eval_env.reset()
                pbar.update(1)
        eval_env.close()
        del eval_env
        mean_eval_rollout_length = np.mean(eval_rollout_lengths)
        std_eval_rollout_length = np.std(eval_rollout_lengths)
        success_rate = np.mean(is_eval_rollout_success)

        self.writer.add_scalar("Evaluation/Mean Rollout Length(Iter)", mean_eval_rollout_length, self.curr_iter_idx)
        self.writer.add_scalar("Evaluation/Mean Rollout Length(Epoch)", mean_eval_rollout_length, self.curr_epoch_idx)
        self.writer.add_scalar("Evaluation/Mean Rollout Length((Numer of Expert Labels)", mean_eval_rollout_length, self.curr_expert_label_num)

        self.writer.add_scalar("Evaluation/Std Rollout Length(Iter)", std_eval_rollout_length, self.curr_iter_idx)
        self.writer.add_scalar("Evaluation/Std Rollout Length(Epoch)", std_eval_rollout_length, self.curr_epoch_idx)
        self.writer.add_scalar("Evaluation/Std Rollout Length(Number of Expert Labels)", std_eval_rollout_length, self.curr_expert_label_num)

        self.writer.add_scalar("Evaluation/Success Rate(Iter)", success_rate, self.curr_iter_idx)
        self.writer.add_scalar("Evaluation/Success Rate(Epoch)", success_rate, self.curr_epoch_idx)
        self.writer.add_scalar("Evaluation/Success Rate(Number of Expert Labels)", success_rate, self.curr_expert_label_num)

        log_imgs = np.stack(log_imgs)
        log_imgs = np.expand_dims(log_imgs, axis=0)  # Add batch dimension
        log_imgs = torch.from_numpy(log_imgs)
        log_imgs = log_imgs.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to (N, T, C, H, W)

        # Reduce the resolution by 4 times for each timestep
        resized_imgs = []
        for i in range(log_imgs.shape[1]):
            resized_img = F.interpolate(log_imgs[:, i], scale_factor=0.4, mode='bilinear', align_corners=False)
            resized_imgs.append(resized_img)
        log_imgs = torch.stack(resized_imgs, dim=1)

        # print("Shape of img tensor after resolution reduction: ", imgs.shape)
        self.writer.add_video('Evaluation/Video(Iter)', log_imgs, self.curr_iter_idx)

@hydra.main(
    version_base=None,
    config_path='../../config',
    config_name="mavis_pick_and_place_bc_dp_unet")
def main(cfg):
    mavis_incremental_bc = MAVISIncrementalBehavioralCloning(cfg)
    mavis_incremental_bc.run()

if __name__ == "__main__":
    main()
