from pathlib import Path
import numpy as np
import zarr

from nn_utils.dataset.base_dataset import BaseDataset

class SingleStepStateOnlyDataset(BaseDataset):
    def __init__(self,
                 dataset_path: str,
                 state_dim: int = 1080,
                 action_dim: int = 2):
        super().__init__(dataset_path=dataset_path,
                         state_dim=state_dim,
                         action_dim=action_dim)

    def __len__(self):
        return self.episode_ends[-1] - 1

    def __getitem__(self, idx):
        file_idx = self.episode_ends.searchsorted(idx, side='right')
        file = self.loaders[file_idx]

        processed_idx = idx - (self.episode_ends[file_idx - 1] if file_idx > 0 else 0)

        sampled_system_states = file[self.state_key][processed_idx]
        sampled_actions = file[self.action_key][processed_idx]

        return sampled_system_states, sampled_actions

    def create_new_rollout_dataset(self):
        super().create_new_rollout_dataset()

    def append_to_curr_rollout_dataset(self, state, action, is_expert_label=True):
        super().append_to_curr_rollout_dataset(state, action, is_expert_label)

    def stop_recording_curr_rollout_dataset_and_add_it_to_this_object(self):
        super().append_to_curr_rollout_dataset()


class RecedingHorizonStateOnlyDataset(BaseDataset):
    def __init__(self,
                 dataset_path: str,
                 state_dim: int = 1080,
                 action_dim: int = 2,
                 pred_horizon: int = 16,
                 obs_horizon: int = 2,
                 action_horizon: int = 8):

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        self.episode_ends_offset_by_pred_horizon = None
        self.episode_begins = None
        self.episode_begins_offset_by_obs_horizon = None

        self.total_num_expert_labels = 0
        self.incremental_num_expert_labels_from_each_rollout = []

        super().__init__(dataset_path=dataset_path,
                         state_dim=state_dim,
                         action_dim=action_dim)

    def __len__(self):
        return self.total_num_expert_labels - 1

    def try_loading_collected_rollouts(self, dataset_path):
        super().try_loading_collected_rollouts(dataset_path)

        self.episode_ends_offset_by_pred_horizon = self.episode_ends - self.pred_horizon

        self.episode_begins = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_begins_offset_by_obs_horizon = self.episode_begins + self.obs_horizon - 1

        self.total_num_expert_labels = 0
        self.incremental_num_expert_labels_from_each_rollout = []
        for loader in self.loaders:
            self.total_num_expert_labels += np.sum(loader["is_expert_label"][:])
            self.incremental_num_expert_labels_from_each_rollout.append(self.total_num_expert_labels)

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

    def convert_expert_label_idx_to_global_idx(self, expert_label_idx, file_idx):
        #print("====================================")
        #print("expert_label_idx: ", expert_label_idx)
        #print("file_idx: ", file_idx)
        # find out this is the ?th expert label in the dataset
        num_expert_labels_from_previous_rollouts = self.incremental_num_expert_labels_from_each_rollout[file_idx - 1] if file_idx > 0 else 0
        expert_label_idx_in_target_rollout = expert_label_idx - num_expert_labels_from_previous_rollouts
        # find out the corresponding idx in the rollout that this ?th expert label belongs to by checking the numberof boolean values in the is_expert_label array
        true_indices = np.where(self.loaders[file_idx]["is_expert_label"][:])[0]
        offset = self.episode_ends[file_idx - 1] if file_idx > 0 else 0
        rollout_step_idx = true_indices[expert_label_idx_in_target_rollout] + offset

        #print("num_expert_labels_from_previous_rollouts: ", num_expert_labels_from_previous_rollouts)

        #print("expert_label_idx_in_target_rollout: ", expert_label_idx_in_target_rollout)
        #print("true_indices: ", true_indices)
        #print("offset: ", offset)
        #print("rollout_step_idx: ", rollout_step_idx)
        #print("====================================")
        return rollout_step_idx

    def __getitem__(self, idx):
        #print("incremental_num_expert_labels_from_each_rollout: ", self.incremental_num_expert_labels_from_each_rollout)
        file_idx = np.array(self.incremental_num_expert_labels_from_each_rollout).searchsorted(idx, side='right')
        file = self.loaders[file_idx]

        idx = self.convert_expert_label_idx_to_global_idx(idx, file_idx)

        is_end_of_episode, num_steps_remaining_to_end = self.check_if_idx_at_end_of_episode(idx, file_idx)
        is_beginning_of_episode, num_steps_remaining_to_start = self.check_if_idx_at_beginning_of_episode(idx, file_idx)

        processed_idx = idx - (self.episode_ends[file_idx - 1] if file_idx > 0 else 0)

        if not is_beginning_of_episode and not is_end_of_episode:
            sampled_system_states = file[self.state_key][processed_idx - self.obs_horizon + 1:processed_idx + 1]
            sampled_actions = file[self.action_key][processed_idx:processed_idx + self.pred_horizon]

        elif is_beginning_of_episode and not is_end_of_episode:
            num_samples_we_have = self.obs_horizon - num_steps_remaining_to_start

            sampled_system_states = np.concatenate(
                [np.tile(file[self.state_key][0], (num_steps_remaining_to_start, 1)),
                 file[self.state_key][:num_samples_we_have]], axis=0)

            sampled_actions = file[self.action_key][processed_idx:processed_idx + self.pred_horizon]

        elif not is_beginning_of_episode and is_end_of_episode:
            num_action_samples_to_pad = self.pred_horizon - num_steps_remaining_to_end

            sampled_system_states = file[self.state_key][processed_idx - self.obs_horizon + 1:processed_idx + 1]
            sampled_actions = np.concatenate([file[self.action_key][processed_idx:processed_idx + num_steps_remaining_to_end],
                                              np.tile(file[self.action_key][-1], (num_action_samples_to_pad, 1))], axis=0)

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

        #print("shape of sampled_system_states: ", sampled_system_states.shape)
        #print("shape of sampled_actions: ", sampled_actions.shape)
        return sampled_system_states, sampled_actions

    def create_new_rollout_dataset(self):
        super().create_new_rollout_dataset()

    def append_to_curr_rollout_dataset(self, state, action, is_expert_label=True):
        super().append_to_curr_rollout_dataset(state, action, is_expert_label)

    def stop_recording_curr_rollout_dataset_and_add_it_to_this_object(self):
        super().stop_recording_curr_rollout_dataset_and_add_it_to_this_object()

        self.episode_ends_offset_by_pred_horizon = self.episode_ends - self.pred_horizon

        self.episode_begins = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_begins_offset_by_obs_horizon = self.episode_begins + self.obs_horizon - 1

        self.total_num_expert_labels += np.sum(self.loaders[-1]["is_expert_label"][:])

        self.incremental_num_expert_labels_from_each_rollout.append(self.total_num_expert_labels)