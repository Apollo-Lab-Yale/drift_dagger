import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from nn_utils.legacy.network import create_nets

def normalize_images(images, rgb_stat):
    nimages = (images - rgb_stat['min']) / (rgb_stat['max'] - rgb_stat['min'])
    nimages = nimages * 2 - 1

    nimages = np.transpose(nimages, (0, 3, 1, 2))

    return nimages

class MAVISLegacyExpertPolicy(nn.Module):
    def __init__(self,
                 ckpt_file_dir,
                 device,
                 vision_feature_dim=512,
                 lowdim_obs_dim=10,
                 img_width=256,
                 img_height = 192,
                 pred_horizon=16,
                 obs_horizon=2,
                 action_horizon=8,
                 action_dim=10,
                 num_diffusion_iters=100
                 ):
        super().__init__()

        self.device = device

        self.nets = create_nets(
            vision_feature_dim=vision_feature_dim,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon
        )

        self.lowdim_obs_dim = lowdim_obs_dim
        self.img_width = img_width
        self.img_height = img_height
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_diffusion_iters = num_diffusion_iters

        self.nets["noise_pred_net"] = self.nets["noise_pred_net"].to(device)
        self.nets["vision_encoder"] = self.nets["vision_encoder"].to(device)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters,
                                             beta_schedule='squaredcos_cap_v2',
                                             clip_sample=True,
                                             prediction_type='epsilon')

        state_dict = torch.load(ckpt_file_dir, map_location=device)

        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        self.nets.load_state_dict(new_state_dict)

        self.rgb_stat = {
            'min': np.tile(np.array([0.0, 0.0, 0.0]),
                           (obs_horizon, img_height, img_width, 1)),
            'max': np.tile(np.array([255.0, 255.0, 255.0]),
                           (obs_horizon, img_height, img_width, 1))
        }

    def forward(self, sample, timestep, global_cond):
        noise_pred = self.nets["noise_pred_net"](sample, timestep, global_cond=global_cond)
        return noise_pred

    def predict_action(self, obs) -> list:
        B = 1

        images, agent_poses = obs

        nagent_poses = agent_poses

        nimages = normalize_images(images, self.rgb_stat)

        nimages = torch.from_numpy(nimages).to(self.device, dtype=torch.float32)
        nagent_poses = torch.from_numpy(nagent_poses).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            image_features = self.nets['vision_encoder'](nimages)

            obs_features = torch.cat([image_features, nagent_poses], dim=-1)

            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            noisy_action = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.forward(sample=naction,
                                          timestep=k,
                                          global_cond=obs_cond)

                naction = self.noise_scheduler.step(
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