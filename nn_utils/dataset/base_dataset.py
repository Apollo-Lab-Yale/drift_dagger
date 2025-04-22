from pathlib import Path
import numpy as np
import zarr

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 state_dim: int,
                 action_dim: int):
        super().__init__()

        self.dataset_path = dataset_path

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_key = "state"
        self.action_key = "action"

        self.curr_sample_idx = 0
        self.curr_rollout_ds = None

        self.loaders = []
        self.episode_ends = np.array([0])

        self.curr_rollout_idx = 0

        self.try_loading_collected_rollouts(dataset_path)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raise NotImplementedError()

    def try_loading_collected_rollouts(self, dataset_path):
        if Path(dataset_path).exists():
            print("====================================")
            print("Found existing rollouts in directory: ", dataset_path)
            data_files = [str(file) for file in Path(dataset_path).glob('[0-9]*')]
            # Sort data_files for consistency
            data_files.sort(key=lambda x: int(Path(x).name))
            # Prepare lazy loaders for data
            self.loaders = [zarr.open(file, mode='r') for file in data_files]

            episode_ends_separated = np.array([len(loader["step_idx"]) for loader in self.loaders])
            self.episode_ends = np.cumsum(episode_ends_separated)

            self.curr_rollout_idx = len(self.loaders)
            print("The number of existing rollouts: ", len(self.loaders))
            print("====================================")

    def create_new_rollout_dataset(self):
        zarr_rollout_dataset_path = f"{self.dataset_path}/{self.curr_rollout_idx}"

        self.curr_rollout_ds = zarr.open(zarr_rollout_dataset_path, mode='a')

        step_idx = self.curr_rollout_ds.create_dataset("step_idx",
                                                       shape=(0,),
                                                       chunks=(1,),
                                                       dtype=np.int64)

        state_rollout_ds = self.curr_rollout_ds.create_dataset(self.state_key,
                                                               shape=(0, self.state_dim),
                                                               chunks=(1, self.state_dim),
                                                               dtype=np.float64)

        action_rollout_ds = self.curr_rollout_ds.create_dataset(self.action_key,
                                                                shape=(0, self.action_dim),
                                                                chunks=(1, self.action_dim),
                                                                dtype=np.float64)

        is_expert_label_ds = self.curr_rollout_ds.create_dataset("is_expert_label",
                                                                 shape=(0,),
                                                                 chunks=(1,),
                                                                 dtype=bool)

    def append_to_curr_rollout_dataset(self, state, action, is_expert_label=True):
        if self.curr_rollout_ds is None:
            raise Exception("No current rollout dataset exists! Please create one first.")
        self.curr_rollout_ds["step_idx"].append(np.array([self.curr_sample_idx]))
        self.curr_rollout_ds[self.state_key].append(np.array(state)[None, ...])
        self.curr_rollout_ds[self.action_key].append(np.array(action)[None, ...])
        self.curr_rollout_ds["is_expert_label"].append(np.array([is_expert_label]))

        self.curr_sample_idx += 1

    def stop_recording_curr_rollout_dataset_and_add_it_to_this_object(self):
        self.curr_rollout_ds = None
        self.loaders.append(zarr.open(f"{self.dataset_path}/{self.curr_rollout_idx}", mode='r'))

        episode_ends_separated = np.array([len(loader["step_idx"]) for loader in self.loaders])
        self.episode_ends = np.cumsum(episode_ends_separated)

        self.curr_rollout_idx += 1

        self.curr_sample_idx = 0
