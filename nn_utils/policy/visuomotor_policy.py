import torch
import numpy as np
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from einops import rearrange, reduce
from nn_utils.model.unet import ConditionalUnet1D
from nn_utils.policy.base_policy import BasePolicy
from nn_utils.model.transformer import TransformerForDiffusion
from nn_utils.model.mask_generator import LowdimMaskGenerator
from nn_utils.model.vision_encoder import get_resnet, replace_bn_with_gn
from nn_utils.model.unet import ConditionalUnet1D, ConditionalResidualBlock1D, Downsample1d, Upsample1d, Conv1dBlock


def normalize_images(images, rgb_stat):
    nimages = (images - rgb_stat['min']) / (rgb_stat['max'] - rgb_stat['min'])
    nimages = nimages * 2 - 1

    nimages = np.transpose(nimages, (0, 3, 1, 2))

    return nimages

class VisuomotorUNetDiffusionPolicy(BasePolicy):
    def __init__(self, state_dim,
                 action_dim,
                 img_width,
                 img_height,
                 vision_feature_dim,
                 down_dims,
                 pred_horizon,
                 obs_horizon,
                 action_horizon,
                 diffusion_step_embed_dim,
                 num_training_diffusion_iters,
                 num_inference_diffusion_iters):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.img_width = img_width
        self.img_height = img_height
        self.vision_feature_dim = vision_feature_dim
        self.obs_dim = state_dim + vision_feature_dim

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_training_diffusion_iters = num_training_diffusion_iters
        self.num_inference_diffusion_iters = num_inference_diffusion_iters

        self.max_rank = 0

        self.training_noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_training_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_inference_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

        vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(vision_encoder)

        self.noise_pred_net = ConditionalUnet1D(input_dim=action_dim,
                                                global_cond_dim=self.obs_dim * obs_horizon,
                                                diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims)

        self.rgb_stat = {'min': np.tile(np.array([0.0, 0.0, 0.0]),
                                   (self.obs_horizon, self.img_height, self.img_width, 1)),
                         'max': np.tile(np.array([255.0, 255.0, 255.0]),
                                   (self.obs_horizon, self.img_height, self.img_width, 1))}

        self.layers_to_exclude_in_noise_pred_net = self.get_layers_to_exclude_in_noise_pred_net()
        self.get_max_rank_of_noise_pred_net()


    def forward(self, sample, timestep, global_cond):
        # global_cond = self.lidar_encoder(global_cond)
        noise_pred = self.noise_pred_net(sample, timestep, global_cond=global_cond)
        return noise_pred

    def predict_action(self, obs) -> list:
        B = 1

        images, agent_poses = obs

        nagent_poses = agent_poses

        nimages = normalize_images(images, self.rgb_stat)

        nimages = torch.from_numpy(nimages).to(self.device, dtype=torch.float32)
        nagent_poses = torch.from_numpy(nagent_poses).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            image_features = self.vision_encoder(nimages)

            obs_features = torch.cat([image_features, nagent_poses], dim=-1)

            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            noisy_action = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            self.inference_noise_scheduler.set_timesteps(self.num_inference_diffusion_iters)

            for k in self.inference_noise_scheduler.timesteps:
                noise_pred = self.forward(sample=naction,
                                          timestep=k,
                                          global_cond=obs_cond)

                naction = self.inference_noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            naction = naction.detach().to('cpu').numpy()
            action_pred = naction[0]

            start = self.obs_horizon - 1
            end = start + self.action_horizon
            action = action_pred[start:end, :]

            return action

    def compute_loss(self, batch):
        nimage, nagent_pos, naction = batch
        B = nagent_pos.shape[0]

        image_features = self.vision_encoder(nimage.flatten(end_dim=1).float())
        image_features = image_features.reshape(*nimage.shape[:2], -1)

        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

        noise = torch.randn(naction.shape, device=self.device)

        timesteps = torch.randint(
            0, self.training_noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        noisy_actions = self.training_noise_scheduler.add_noise(naction, noise, timesteps)

        noisy_actions = noisy_actions.float()
        timesteps = timesteps.float()
        obs_cond = obs_cond.float()

        noise_pred = self.forward(noisy_actions, timesteps, obs_cond)

        #print(f"Outputs require_grad: {noise_pred.requires_grad}")

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss

    def is_first_conv1d_block_in_down_modules(self, module_name):
        return module_name.startswith('down_modules.0')

    def get_max_rank_of_noise_pred_net(self):
        return self.get_max_rank_of_network(self.noise_pred_net, layers_to_exclude=self.layers_to_exclude_in_noise_pred_net)

    def get_max_rank_of_network(self, network, layers_to_exclude=[]):
        for name, module in network.named_children():
            # check if it is excluded
            module_full_name = f"{network.__class__.__name__}.{name}"
            #print("layers_to_exclude:", layers_to_exclude)
            if module_full_name in layers_to_exclude:
                #print("module_full_name:", module_full_name)
                continue
            if isinstance(module, nn.Conv1d):
                curr_max_rank = min(module.in_channels, module.out_channels)
                # print("curr_max_rank:", curr_max_rank)
                if curr_max_rank > self.max_rank:
                    self.max_rank = curr_max_rank
            else:
                self.get_max_rank_of_network(module, layers_to_exclude=layers_to_exclude)

    def get_layers_to_exclude_in_noise_pred_net(self):
        layers_to_exclude = []
        network = self.noise_pred_net
        for name, module in network.named_modules():
            if isinstance(module, ConditionalResidualBlock1D):
                # Exclude residual_conv and cond_encoder in ConditionalResidualBlock1D
                layers_to_exclude.append(f"{name}.residual_conv")
                layers_to_exclude.append(f"{name}.cond_encoder")
            elif isinstance(module, (Downsample1d, Upsample1d)):
                # Exclude Downsample1d and Upsample1d layers
                layers_to_exclude.append(name)
            elif isinstance(module, Conv1dBlock):
                # Optionally exclude the first Conv1dBlock in down_modules
                if self.is_first_conv1d_block_in_down_modules(name):
                    layers_to_exclude.append(name)
            elif name == 'final_conv':
                # Exclude final_conv
                layers_to_exclude.append(name)
        return layers_to_exclude


class VisuomotorTransformerDiffusionPolicy(BasePolicy):
    def __init__(self,
                 state_dim,
                 action_dim,
                 img_width,
                 img_height,
                 vision_feature_dim,
                 n_layer,
                 n_head,
                 n_emb,
                 p_drop_emb,
                 p_drop_attn,
                 causal_attn,
                 time_as_cond,
                 n_cond_layers,
                 pred_horizon,
                 obs_horizon,
                 action_horizon,
                 num_training_diffusion_iters):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.img_width = img_width
        self.img_height = img_height
        self.vision_feature_dim = vision_feature_dim
        self.obs_dim = state_dim + vision_feature_dim

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_training_diffusion_iters = num_training_diffusion_iters

        vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(vision_encoder)

        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=pred_horizon,
            n_obs_steps=obs_horizon,
            cond_dim=self.obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_training_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=obs_horizon,
            fix_obs_steps=True,
            action_visible=False
        )

        self.rgb_stat = {'min': np.tile(np.array([0.0, 0.0, 0.0]),
                                   (self.obs_horizon, self.img_height, self.img_width, 1)),
                         'max': np.tile(np.array([255.0, 255.0, 255.0]),
                                   (self.obs_horizon, self.img_height, self.img_width, 1))}

    def forward(self, trajectory, timestep, cond):
        return self.model(trajectory, timestep, cond)

    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None
            ):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.learner_device)

        self.noise_scheduler.set_timesteps(self.num_training_diffusion_iters)

        for t in self.noise_scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            #print("shape of trajectory for conditional sample: ", trajectory.shape)
            #print("shape of condition_data for conditional sample: ", condition_data.shape)
            #print("shape of condition_mask for conditional sample: ", condition_mask.shape)
            #print("shape of cond for conditional sample: ", cond.shape)

            # 2. predict model output
            model_output = self.model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory,
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs):
        B = 1
        _, Do = obs.shape
        To = self.obs_horizon
        assert Do == self.state_dim
        T = self.pred_horizon
        Da = self.action_dim

        # Build input
        device = self.device
        dtype = self.dtype

        cond = None
        cond_data = None
        cond_mask = None
        # Pass obs as cond
        cond = obs[:To, :]
        cond = cond.unsqueeze(0).expand(B, To, Do)
        shape = (B, T, Da)
        #if self.pred_action_steps_only:
        #    shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        #print("shape of cond_data for pred action: ", cond_data.shape)
        #print("shape of cond_mask for pred action: ", cond_mask.shape)
        #print("shape of cond for pred action: ", cond.shape)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond)

        naction = nsample.detach().to('cpu').numpy()
        action_pred = naction[0]

        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]

        return action

    def compute_loss(self, batch):
        obs, action = batch

        cond = None
        trajectory = action

        # Handle obs as cond
        cond = obs[:, :self.obs_horizon, :]

        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.learner_device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.learner_device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        #print("shape of noisy_trajectory for compute loss: ", noisy_trajectory.shape)
        #print("shape of trajectory for compute loss: ", trajectory.shape)
        #print("shape of condition_mask for compute loss: ", condition_mask.shape)
        #print("shape of cond for compute loss: ", cond.shape)

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        target = noise

        loss = torch.nn.functional.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss