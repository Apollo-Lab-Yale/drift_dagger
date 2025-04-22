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
from diffusers.optimization import get_scheduler
from nn_utils.policy.visuomotor_policy import (VisuomotorUNetDiffusionPolicy,
                                               VisuomotorTransformerDiffusionPolicy)

from nn_utils.model.rank_scheduler import RankScheduler

from nn_utils.policy.visuomotor_policy_with_lora import (VisuomotorUNetDiffusionPolicyWithLoRA,
                                                         VisuomotorTransformerDiffusionPolicyWithLoRA)

from nn_utils.policy.visuomotor_policy_with_rank_modulation import VisuomotorUNetDiffusionPolicyWithRankModulation

from nn_utils.dataset.visuomotor_dataset import RecedingHorizonVisuomotorDataset



class BaseOnlineTraining:
    def __init__(self, cfg):
        self.cfg = cfg
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.learner_device = torch.device("cuda:1")
            self.expert_device = torch.device("cuda:0")
        else:
            raise ValueError("The script should be run on a machine with at least two CUDA GPUs")

        self.log_dir = f"logs/{self.cfg.training_session_name}"
        self.ckpt_dir = f"ckpts/{self.cfg.training_session_name}"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.num_offline_rollouts_for_bootstrapping = 0
        if not self.cfg.use_offline_bootstrapping_dataset:
            self.dataset_dir = f"demonstrations/{self.cfg.training_session_name}"
            os.makedirs(self.dataset_dir, exist_ok=True)
        else:
            self.dataset_dir = self.cfg.offline_bootstrapping_dataset_path
            # count the number of subdirectories in the dataset directory
            self.offline_bootstrapping_subfolders = [f.path for f in os.scandir(self.dataset_dir) if f.is_dir()]
            self.num_offline_rollouts_for_bootstrapping = len(self.offline_bootstrapping_subfolders)
            print("Use offline bootstrapping dataset with {} rollouts".format(self.num_offline_rollouts_for_bootstrapping))

        self.writer = SummaryWriter(self.log_dir)

        seed = self.cfg.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.learner = BasePolicy()

        self.dataset = BaseDataset(dataset_path=self.dataset_dir,
                                   state_dim=self.cfg.state_dim,
                                   action_dim=self.cfg.action_dim)

        self.dataloader = None

        self.curr_epoch_idx = 0
        self.curr_rollout_idx = 0
        self.curr_iter_idx = 0
        self.curr_expert_label_num = 0

        self.finished_bootstrapping = False

        self.max_num_training_steps = ((self.cfg.num_iter * self.cfg.num_epoch_per_iter) *
                                       (self.cfg.num_iter * self.cfg.num_rollout_per_iter *
                                        self.cfg.max_num_steps_per_rollout / self.cfg.dataloader.batch_size))

        if not self.cfg.use_low_rank_adapter and not self.cfg.use_rank_modulation:
            if self.cfg.network.type == "dp_unet":
                self.learner = VisuomotorUNetDiffusionPolicy(state_dim=self.cfg.state_dim,
                                                             action_dim=self.cfg.state_dim,
                                                             img_width=self.cfg.downsampled_img_width,
                                                             img_height=self.cfg.downsampled_img_height,
                                                             vision_feature_dim=self.cfg.network.vision_feature_dim,
                                                             down_dims=self.cfg.network.down_dims,
                                                             pred_horizon=self.cfg.network.pred_horizon,
                                                             obs_horizon=self.cfg.network.obs_horizon,
                                                             action_horizon=self.cfg.network.action_horizon,
                                                             diffusion_step_embed_dim=self.cfg.network.diffusion_step_embed_dim,
                                                             num_training_diffusion_iters=self.cfg.network.num_training_diffusion_iters,
                                                             num_inference_diffusion_iters=self.cfg.network.num_inference_diffusion_iters)
            elif self.cfg.network.type == "dp_transformer":
                # TODO: Implement transformer backbone for visuomotor DP
                pass
            else:
                raise ValueError(f"Invalid network type: {self.cfg.network.type}")

            self.learner.to(self.learner_device)
        elif self.cfg.use_low_rank_adapter or self.cfg.use_rank_modulation:
            if self.cfg.use_low_rank_adapter:
                if self.cfg.network.type == "dp_unet":
                    self.learner = VisuomotorUNetDiffusionPolicyWithLoRA(state_dim=self.cfg.state_dim,
                                                                         action_dim=self.cfg.state_dim,
                                                                         img_width=self.cfg.downsampled_img_width,
                                                                         img_height=self.cfg.downsampled_img_height,
                                                                         vision_feature_dim=self.cfg.network.vision_feature_dim,
                                                                         down_dims=self.cfg.network.down_dims,
                                                                         pred_horizon=self.cfg.network.pred_horizon,
                                                                         obs_horizon=self.cfg.network.obs_horizon,
                                                                         action_horizon=self.cfg.network.action_horizon,
                                                                         diffusion_step_embed_dim=self.cfg.network.diffusion_step_embed_dim,
                                                                         num_training_diffusion_iters=self.cfg.network.num_training_diffusion_iters,
                                                                         num_inference_diffusion_iters=self.cfg.network.num_inference_diffusion_iters,
                                                                         lora_dropout_p=self.cfg.network.lora_dropout_p,
                                                                         lora_scale=self.cfg.network.lora_scale,
                                                                         exclude_critical_layers_from_lora=self.cfg.network.exclude_critical_layers_from_lora)
                elif self.cfg.network.type == "dp_transformer":
                    # TODO: Implement transformer backbone for visuomotor DP
                    pass
                else:
                    raise ValueError(f"Invalid network type: {self.cfg.network.type}")

                self.learner.to(self.learner_device)

                if self.cfg.rank_scheduler.use_rank_scheduler:
                    max_rank = self.learner.max_rank
                    self.rank_scheduler = RankScheduler(r_max=max_rank,
                                                        r_min=self.cfg.rank_scheduler.r_min,
                                                        num_training_steps=self.cfg.num_bootstrapping_epochs,
                                                        decay_type=self.cfg.rank_scheduler.decay_type)
                    self.curr_low_rank = self.rank_scheduler.get_rank(**self.cfg.rank_scheduler)

                if not self.cfg.rank_scheduler.use_rank_scheduler:
                    self.curr_low_rank = self.cfg.network.fixed_lora_rank


                if self.cfg.network.inject_low_rank_adapter_at_init:
                    if not self.cfg.use_pretrained_policy:
                        self.inject_low_rank_adapter_to_learner()
                    else:
                        print("Due to loading pretrained policy, the low-rank adapter cannot be injected at init.")
            elif self.cfg.use_rank_modulation:
                if self.cfg.network.type == "dp_unet":
                    self.learner = VisuomotorUNetDiffusionPolicyWithRankModulation(state_dim=self.cfg.state_dim,
                                                                                 action_dim=self.cfg.state_dim,
                                                                                 img_width=self.cfg.downsampled_img_width,
                                                                                 img_height=self.cfg.downsampled_img_height,
                                                                                 vision_feature_dim=self.cfg.network.vision_feature_dim,
                                                                                 down_dims=self.cfg.network.down_dims,
                                                                                 pred_horizon=self.cfg.network.pred_horizon,
                                                                                 obs_horizon=self.cfg.network.obs_horizon,
                                                                                 action_horizon=self.cfg.network.action_horizon,
                                                                                 diffusion_step_embed_dim=self.cfg.network.diffusion_step_embed_dim,
                                                                                 num_training_diffusion_iters=self.cfg.network.num_training_diffusion_iters,
                                                                                 num_inference_diffusion_iters=self.cfg.network.num_inference_diffusion_iters,)
                elif self.cfg.network.type == "dp_transformer":
                    pass
                else:
                    raise ValueError(f"Invalid network type: {self.cfg.network.type}")

                self.learner.to(self.learner_device)

                if self.cfg.rank_scheduler.use_rank_scheduler:
                    max_rank = self.learner.max_rank
                    self.rank_scheduler = RankScheduler(r_max=max_rank,
                                                        r_min=self.cfg.rank_scheduler.r_min,
                                                        num_training_steps=self.cfg.num_bootstrapping_epochs,
                                                        decay_type=self.cfg.rank_scheduler.decay_type)
                    self.curr_low_rank = self.rank_scheduler.get_rank(**self.cfg.rank_scheduler)
            else:
                raise ValueError("Cannot use both low-rank adapter and rank modulation at the same time.")






        self.dataset = RecedingHorizonVisuomotorDataset(dataset_path=self.dataset_dir,
                                                        state_dim=self.cfg.state_dim,
                                                        action_dim=self.cfg.action_dim,
                                                        img_width=self.cfg.downsampled_img_width,
                                                        img_height=self.cfg.downsampled_img_height,
                                                        pred_horizon=self.cfg.network.pred_horizon,
                                                        obs_horizon=self.cfg.network.obs_horizon,
                                                        action_horizon=self.cfg.network.action_horizon)

        self.optimizer = torch.optim.Adam(params=self.learner.parameters(),
                                          lr=self.cfg.optimizer.learning_rate,
                                          weight_decay=self.cfg.optimizer.weight_decay)

        self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler,
                                          optimizer=self.optimizer,
                                          num_warmup_steps=self.cfg.lr_warmup_steps,
                                          num_training_steps=self.max_num_training_steps
                                          )

        self.curr_step_idx = 0

    def inject_low_rank_adapter_to_learner(self):
        if not self.learner.lora_injected:
            self.learner.inject_lora(self.cfg.network.apply_lora_to_visual_encoder, self.curr_low_rank)

    def training_one_batch(self, batch_idx, batch):
        return 0

    def collect_one_rollout(self, interactive=False):
        return False

    def freeze_main_weights_and_update_optimizer_and_scheduler(self):
        if self.cfg.network.completely_freeze_main_params:
            self.learner.freeze_main_network(self.cfg.network.apply_lora_to_visual_encoder)
            self.update_optimizer_parameters()
            self.update_scheduler()
        else:
            main_params, lora_params = self.learner.get_main_and_lora_params()

            # Update the optimizer with parameter groups
            self.optimizer = torch.optim.Adam([
                {'params': lora_params, 'lr': self.cfg.optimizer.learning_rate},
                {'params': main_params, 'lr': 0.0}
            ], weight_decay=self.cfg.optimizer.weight_decay)

            # Update the scheduler
            self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler,
                                              optimizer=self.optimizer,
                                              num_warmup_steps=self.cfg.lr_warmup_steps,
                                              num_training_steps=(
                                                      (self.cfg.num_iter * self.cfg.num_epoch_per_iter) *
                                                      (self.cfg.num_iter * self.cfg.num_rollout_per_iter *
                                                       self.cfg.max_num_steps_per_rollout / self.cfg.dataloader.batch_size)
                                              ))

    def update_optimizer_parameters(self):
        # Get the list of parameters that require gradients
        trainable_params = list(filter(lambda p: p.requires_grad, self.learner.parameters()))
        trainable_params_set = set(trainable_params)

        # Update optimizer's parameter groups
        optimizer_param_set = set()
        for group in self.optimizer.param_groups:
            optimizer_param_set.update(group['params'])

        # Identify new parameters to add to the optimizer
        new_params = [p for p in trainable_params if p not in optimizer_param_set]

        # Remove parameters that no longer require gradients from optimizer param groups
        for group in self.optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad and p in trainable_params_set]

        # Add new parameters to the optimizer
        if new_params:
            self.optimizer.add_param_group({'params': new_params, 'lr': self.cfg.optimizer.learning_rate})

        # Remove state for parameters that are no longer present
        optimizer_state_params = set(self.optimizer.state.keys())
        params_to_remove = optimizer_state_params - trainable_params_set
        for param in params_to_remove:
            del self.optimizer.state[param]

    def update_scheduler(self):
        scheduler_state = self.lr_scheduler.state_dict()
        self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler,
                                          optimizer=self.optimizer,
                                          num_warmup_steps=self.cfg.lr_warmup_steps,
                                          num_training_steps=(
                                              (self.cfg.num_iter * self.cfg.num_epoch_per_iter) *
                                              (self.cfg.num_iter * self.cfg.num_rollout_per_iter *
                                               self.cfg.max_num_steps_per_rollout / self.cfg.dataloader.batch_size)
                                            ))
        self.lr_scheduler.load_state_dict(scheduler_state)

    def training_one_epoch(self):
        if self.dataloader is None:
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.cfg.dataloader.batch_size,
                                         shuffle=self.cfg.dataloader.shuffle,
                                         num_workers=self.cfg.dataloader.num_workers)
        losses = []

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Batch Progress")):
            loss = self.training_one_batch(batch_idx, batch)
            losses.append(loss)

            self.writer.add_scalar('Training/Loss(Step)', loss, self.curr_step_idx)
            self.curr_step_idx += 1

        mean_loss = np.mean(losses)

        self.writer.add_scalar("Training/Loss(Epoch)", mean_loss, self.curr_epoch_idx)

        # number of trainable parameters in learner
        num_params = sum(p.numel() for p in self.learner.noise_pred_net.parameters() if p.requires_grad)
        self.writer.add_scalar("Training/Number of Trainable Parameters", num_params, self.curr_epoch_idx)

        if self.cfg.rank_scheduler.use_rank_scheduler and (self.cfg.use_rank_modulation or self.cfg.use_low_rank_adapter):
            self.rank_scheduler.step_increment()

            new_rank = self.rank_scheduler.get_rank(**self.cfg.rank_scheduler)
            if new_rank != self.curr_low_rank:
                self.curr_low_rank = new_rank
                self.learner.reduce_rank(new_rank)

                if self.cfg.use_low_rank_adapter:
                    self.freeze_main_weights_and_update_optimizer_and_scheduler()

                if self.cfg.use_rank_modulation:
                    self.update_optimizer_parameters()
                    self.update_scheduler()

                self.writer.add_scalar('Training/Rank Number(Epoch)', self.curr_low_rank, self.curr_epoch_idx)

        return mean_loss


    def training_one_iter(self):
        mean_losses = []
        with tqdm(range(self.cfg.num_rollout_per_iter), desc="Online Learning: Rollout Collection") as pbar:
            for curr_rollout_idx in pbar:
                res = False
                while not res:
                    res = self.collect_one_rollout(interactive=True)
                self.curr_rollout_idx += 1

        with tqdm(range(self.cfg.num_epoch_per_iter), desc="Online Learning: Epoch Training") as pbar:
            for curr_epoch_idx in pbar:
                mean_loss = self.training_one_epoch()
                self.curr_epoch_idx += 1
                mean_losses.append(mean_loss)

        mean_loss = np.mean(mean_losses)
        self.writer.add_scalar("Training/Loss(Numer of Expert Labels)", mean_loss, self.curr_expert_label_num)
        self.writer.add_scalar("Training/Loss(Iter)", mean_loss, self.curr_iter_idx)

        self.curr_iter_idx += 1

    def bootstrapping(self):
        if not self.cfg.use_pretrained_policy:
            mean_losses = []

            if not self.cfg.use_offline_bootstrapping_dataset:
                with tqdm(range(self.cfg.num_bootstrapping_rollouts), desc="Offline Bootstrapping: Rollout Collection") as pbar:
                    for curr_rollout_idx in pbar:
                        self.collect_one_rollout(interactive=False)
                        self.curr_rollout_idx += 1
            else:
                self.curr_rollout_idx = self.num_offline_rollouts_for_bootstrapping - 1

            with tqdm(range(self.cfg.num_bootstrapping_epochs), desc="Offline Bootstrapping: Epoch Training") as pbar:
                for curr_epoch_idx in pbar:
                    mean_loss = self.training_one_epoch()
                    self.curr_epoch_idx += 1
                    self.curr_iter_idx += 1
                    mean_losses.append(mean_loss)

                    if self.cfg.use_eval and (curr_epoch_idx % self.cfg.eval_and_save_every_n_iters == 0) and self.cfg.eval_when_bootstrapping:
                        self.evaluation()
                        self.learner.save(path=f"{self.ckpt_dir}/{curr_epoch_idx}.ckpt")

            mean_loss = np.mean(mean_losses)
            self.writer.add_scalar("Training/Loss(Numer of Expert Labels)", mean_loss, self.curr_expert_label_num)
            self.writer.add_scalar("Training/Loss(Iter)", mean_loss, self.curr_iter_idx)
        else:
            self.learner.load(path=self.cfg.pretrained_policy_path)

        self.finished_bootstrapping = True

        if self.cfg.use_low_rank_adapter:
            if not self.cfg.rank_scheduler.use_rank_scheduler:
                if self.cfg.network.inject_low_rank_adapter_at_init:
                    if self.cfg.use_pretrained_policy:
                        print("Injecting low-rank adapter to the learner after bootstrapping.")
                        self.inject_low_rank_adapter_to_learner()

                if not self.cfg.network.inject_low_rank_adapter_at_init:
                    self.inject_low_rank_adapter_to_learner()

    def evaluation(self):
        raise NotImplementedError()

    def remove_online_rollouts_from_dataset(self):
        # removing all the online rollouts from the dataset
        # only keep subfolders in offline_bootstrapping_subfolders
        online_rollout_subfolders = [f.path for f in os.scandir(self.dataset_dir) if f.is_dir()]
        for subfolder in online_rollout_subfolders:
            if subfolder not in self.offline_bootstrapping_subfolders:
                os.system(f"rm -rf {subfolder}")

    def run(self):
        self.bootstrapping()

        with tqdm(range(self.cfg.num_iter), desc="Iterations") as pbar:
            for curr_iter_idx in pbar:
                self.training_one_iter()

                if self.cfg.use_eval and (curr_iter_idx % self.cfg.eval_and_save_every_n_iters == 0):
                    self.evaluation()
                    self.learner.save(path=f"{self.ckpt_dir}/{curr_iter_idx}.ckpt")

        if self.cfg.use_offline_bootstrapping_dataset:
            # Clean up offline dataset for next trials
            self.remove_online_rollouts_from_dataset()
