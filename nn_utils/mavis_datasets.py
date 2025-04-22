from pathlib import Path
import numpy as np
import zarr
from copy import deepcopy
import torch
import time
from time import sleep
from torch.utils.data import Dataset

class DiffusionPolicyVisionDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 state_key="lookat_space_euler_state_obs",
                 action_key="lookat_space_euler_action",
                 reward_key="step_rewards",
                 normalize_rgb_images=True,
                 ):
        super().__init__()

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        self.state_key = state_key
        self.action_key = action_key
        self.reward_key = reward_key

        self.normalize_rgb_images = normalize_rgb_images

        data_files = [str(file) for file in Path(dataset_path).glob('[0-9]*')]
        data_files.sort(key=lambda x: int(Path(x).name))
        self.loaders = [zarr.open(file, mode='r') for file in data_files]

        episode_ends_separated = np.array([len(loader["step_idx"]) for loader in self.loaders])
        self.episode_ends = np.cumsum(episode_ends_separated)
        self.episode_ends_offset_by_pred_horizon = self.episode_ends - self.pred_horizon

        self.episode_begins = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_begins_offset_by_obs_horizon = self.episode_begins + self.obs_horizon - 1

        self.img_height = self.loaders[0]['rgb_images'][0].shape[0]
        self.img_width = self.loaders[0]['rgb_images'][0].shape[1]

    def __len__(self):
        return self.episode_ends[-1] - 1

    def normalize_sampled_rgb_images(self, rgb_images_seq):
        rgb_stat = {
                'min': np.tile(np.array([0.0, 0.0, 0.0]), (self.obs_horizon, self.img_height, self.img_width, 1)),
                'max': np.tile(np.array([255.0, 255.0, 255.0]), (self.obs_horizon, self.img_height, self.img_width, 1))
            }
        # Normalize
        rgb_images_seq = (rgb_images_seq - rgb_stat['min']) / (rgb_stat['max'] - rgb_stat['min'])
        # Scale to [-1, 1]
        rgb_images_seq = rgb_images_seq * 2 - 1
        return rgb_images_seq

    def check_if_idx_at_end_of_episode(self, idx, file_idx):
        curr_episode_end_offset_by_pred_horizon = self.episode_ends_offset_by_pred_horizon[file_idx]
        is_end_of_episode = idx >= curr_episode_end_offset_by_pred_horizon
        num_steps_remaining = self.episode_ends[file_idx] - idx
        return is_end_of_episode, num_steps_remaining

    def check_if_idx_at_beginning_of_episode(self, idx, file_idx):
        curr_episode_begin_offset_by_obs_horizon = self.episode_begins_offset_by_obs_horizon[file_idx]
        is_beginning_of_episode = idx < curr_episode_begin_offset_by_obs_horizon
        num_steps_remaining = curr_episode_begin_offset_by_obs_horizon - idx
        return is_beginning_of_episode, num_steps_remaining

    def __getitem__(self, idx):
        file_idx = self.episode_ends.searchsorted(idx, side='right')
        file = self.loaders[file_idx]

        is_end_of_episode, num_steps_remaining_to_end = self.check_if_idx_at_end_of_episode(idx, file_idx)
        is_beginning_of_episode, num_steps_remaining_to_start = self.check_if_idx_at_beginning_of_episode(idx, file_idx)

        processed_idx = idx - (self.episode_ends[file_idx - 1] if file_idx > 0 else 0)

        if not is_beginning_of_episode and not is_end_of_episode:
            sampled_rgb_images = file['rgb_images'][processed_idx - self.obs_horizon + 1:processed_idx + 1]
            sampled_system_states = file[self.state_key][processed_idx - self.obs_horizon + 1:processed_idx + 1]
            sampled_actions = file[self.action_key][processed_idx:processed_idx + self.pred_horizon]
            sampled_reward = file[self.reward_key][processed_idx]  # TODO: This may need to be adjusted

        elif is_beginning_of_episode and not is_end_of_episode:
            num_samples_we_have = self.obs_horizon - num_steps_remaining_to_start

            sampled_rgb_images = np.concatenate(
                [np.tile(file['rgb_images'][0], (num_steps_remaining_to_start, 1, 1, 1)),
                 file['rgb_images'][:num_samples_we_have]], axis=0)

            sampled_system_states = np.concatenate(
                [np.tile(file[self.state_key][0], (num_steps_remaining_to_start, 1)),
                 file[self.state_key][:num_samples_we_have]], axis=0)

            sampled_actions = file[self.action_key][processed_idx:processed_idx + self.pred_horizon]

            sampled_reward = file[self.reward_key][processed_idx]  # TODO: This may need to be adjusted

        elif not is_beginning_of_episode and is_end_of_episode:
            num_action_samples_to_pad = self.pred_horizon - num_steps_remaining_to_end

            sampled_rgb_images = file['rgb_images'][processed_idx - self.obs_horizon + 1:processed_idx + 1]

            sampled_system_states = file[self.state_key][processed_idx - self.obs_horizon + 1:processed_idx + 1]

            sampled_actions = np.concatenate([file[self.action_key][processed_idx:processed_idx + num_steps_remaining_to_end],
                                              np.tile(file[self.action_key][-1], (num_action_samples_to_pad, 1))], axis=0)

            sampled_reward = file[self.reward_key][processed_idx]  # TODO: This may need to be adjusted
        else:
            print("====================================")
            print("Encounter error for index: ", idx)
            print("Printing out relevant information for debugging")
            print("File index: ", file_idx)
            print("Searchsorted value in episode ends: ", self.episode_ends.searchsorted(idx, side='right'))
            print("Corresponding value in episode ends: ", self.episode_ends[file_idx])
            print("Processed index: ", processed_idx)
            print("Is beginning of episode: ", is_beginning_of_episode)
            print("Is end of episode: ", is_end_of_episode)
            print("Number of steps remaining to start: ", num_steps_remaining_to_start)
            print("Number of steps remaining to end: ", num_steps_remaining_to_end)
            print("====================================")
            raise Exception(
                f'Error: Index {idx} is at the beginning of the episode ({is_beginning_of_episode}), and at the end of the episode ({is_end_of_episode})')

        if self.normalize_rgb_images:
            sampled_rgb_images = self.normalize_sampled_rgb_images(sampled_rgb_images)

        if sampled_rgb_images.shape[3] == 3:
            sampled_rgb_images = np.transpose(sampled_rgb_images, (0, 3, 1, 2))

        return sampled_rgb_images, sampled_system_states, sampled_actions, sampled_reward


class DiffusionPolicyStateOnlyDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 state_key="lookat_space_euler_state_obs",
                 action_key="lookat_space_euler_action",
                 reward_key="step_rewards",
                 ):
        super().__init__()

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        self.state_key = state_key
        self.action_key = action_key
        self.reward_key = reward_key

        data_files = [str(file) for file in Path(dataset_path).glob('[0-9]*')]
        # Sort data_files for consistency
        data_files.sort(key=lambda x: int(Path(x).name))
        # Prepare lazy loaders for data
        self.loaders = [zarr.open(file, mode='r') for file in data_files]

        episode_ends_separated = np.array([len(loader["step_idx"]) for loader in self.loaders])
        self.episode_ends = np.cumsum(episode_ends_separated)
        self.episode_ends_offset_by_pred_horizon = self.episode_ends - self.pred_horizon

        self.episode_begins = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_begins_offset_by_obs_horizon = self.episode_begins + self.obs_horizon - 1

    def __len__(self):
        return self.episode_ends[-1] - 1

    def check_if_idx_at_end_of_episode(self, idx, file_idx):
        curr_episode_end_offset_by_pred_horizon = self.episode_ends_offset_by_pred_horizon[file_idx]
        is_end_of_episode = idx >= curr_episode_end_offset_by_pred_horizon
        num_steps_remaining = self.episode_ends[file_idx] - idx
        return is_end_of_episode, num_steps_remaining

    def check_if_idx_at_beginning_of_episode(self, idx, file_idx):
        curr_episode_begin_offset_by_obs_horizon = self.episode_begins_offset_by_obs_horizon[file_idx]
        is_beginning_of_episode = idx < curr_episode_begin_offset_by_obs_horizon
        num_steps_remaining = curr_episode_begin_offset_by_obs_horizon - idx
        return is_beginning_of_episode, num_steps_remaining

    def __getitem__(self, idx):
        file_idx = self.episode_ends.searchsorted(idx, side='right')
        file = self.loaders[file_idx]

        is_end_of_episode, num_steps_remaining_to_end = self.check_if_idx_at_end_of_episode(idx, file_idx)
        is_beginning_of_episode, num_steps_remaining_to_start = self.check_if_idx_at_beginning_of_episode(idx, file_idx)

        processed_idx = idx - (self.episode_ends[file_idx - 1] if file_idx > 0 else 0)

        if not is_beginning_of_episode and not is_end_of_episode:
            sampled_system_states = file[self.state_key][processed_idx - self.obs_horizon + 1:processed_idx + 1]
            sampled_actions = file[self.action_key][processed_idx:processed_idx + self.pred_horizon]
            sampled_reward = file[self.reward_key][processed_idx]  # TODO: This may need to be adjusted

        elif is_beginning_of_episode and not is_end_of_episode:
            num_samples_we_have = self.obs_horizon - num_steps_remaining_to_start

            sampled_system_states = np.concatenate(
                [np.tile(file[self.state_key][0], (num_steps_remaining_to_start, 1)),
                 file[self.state_key][:num_samples_we_have]], axis=0)

            sampled_actions = file[self.action_key][processed_idx:processed_idx + self.pred_horizon]

            sampled_reward = file[self.reward_key][processed_idx]  # TODO: This may need to be adjusted

        elif not is_beginning_of_episode and is_end_of_episode:
            num_action_samples_to_pad = self.pred_horizon - num_steps_remaining_to_end

            sampled_system_states = file[self.state_key][processed_idx - self.obs_horizon + 1:processed_idx + 1]
            sampled_actions = np.concatenate([file[self.action_key][processed_idx:processed_idx + num_steps_remaining_to_end],
                                              np.tile(file[self.action_key][-1], (num_action_samples_to_pad, 1))], axis=0)

            sampled_reward = file[self.reward_key][processed_idx]  # TODO: This may need to be adjusted
        else:
            print("====================================")
            print("Encounter error for index: ", idx)
            print("Printing out relevant information for debugging")
            print("File index: ", file_idx)
            print("Searchsorted value in episode ends: ", self.episode_ends.searchsorted(idx, side='right'))
            print("Corresponding value in episode ends: ", self.episode_ends[file_idx])
            print("Processed index: ", processed_idx)
            print("Is beginning of episode: ", is_beginning_of_episode)
            print("Is end of episode: ", is_end_of_episode)
            print("Number of steps remaining to start: ", num_steps_remaining_to_start)
            print("Number of steps remaining to end: ", num_steps_remaining_to_end)
            print("====================================")
            raise Exception(
                f'Error: Index {idx} is at the beginning of the episode ({is_beginning_of_episode}), and at the end of the episode ({is_end_of_episode})')

        return sampled_system_states, sampled_actions, sampled_reward