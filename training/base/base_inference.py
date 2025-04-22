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

class BaseInference:
    def __init__(self, cfg: OmegaConf, folder_name, ckpt_idx: int):
        self.cfg = cfg

        seed = self.cfg.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_ckpt_path = f"ckpts/{folder_name}/{ckpt_idx}.ckpt"


    def run(self):
        raise NotImplementedError()