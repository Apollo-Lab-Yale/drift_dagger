import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, reduce
from nn_utils.model.unet import ConditionalUnet1D
from nn_utils.policy.base_policy import BasePolicy
from nn_utils.model.transformer import TransformerForDiffusion
from nn_utils.model.mask_generator import LowdimMaskGenerator


class StateOnlyMultiLayerPerceptronPolicy(BasePolicy):
    def __init__(self, state_dim=1080,
                 action_dim=2,
                 hidden_dim=[512, 512, 512], ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Mish(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Mish(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.Mish(),
            nn.Linear(hidden_dim[2], action_dim)
        )

    def forward(self, obs):
        return self.model(obs)

    def predict_action(self, obs) -> list:
        action = self.forward(obs)
        return action.detach().cpu().numpy()

    def compute_loss(self, batch):
        states, actions = batch
        pred_actions = self.forward(states)
        loss = nn.functional.mse_loss(pred_actions, actions)
        return loss


class StateOnlyUNetDiffusionPolicy(BasePolicy):
    def __init__(self, state_dim=1080,
                 action_dim=2,
                 down_dims=[512, 256, 128],
                 pred_horizon=4,
                 obs_horizon=2,
                 action_horizon=2,
                 diffusion_step_embed_dim=512,
                 num_diffusion_iters=100):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.noise_pred_net = ConditionalUnet1D(input_dim=action_dim,
                                                global_cond_dim=state_dim * obs_horizon,
                                                diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims)



    def forward(self, sample, timestep, global_cond):
        # global_cond = self.lidar_encoder(global_cond)
        noise_pred = self.noise_pred_net(sample, timestep, global_cond=global_cond)
        return noise_pred

    def predict_action(self, obs) -> list:
        B = 1

        with torch.no_grad():
            obs_cond = obs.unsqueeze(0).flatten(start_dim=1).float().cuda()

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

    def compute_loss(self, batch):
        states, actions = batch
        B = states.shape[0]

        obs_cond = states[:, :self.obs_horizon, :]
        obs_cond = obs_cond.flatten(start_dim=1)
        noise = torch.randn(actions.shape, device=self.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        noisy_actions = noisy_actions.float()
        timesteps = timesteps.float()
        obs_cond = obs_cond.float()

        noise_pred = self.forward(noisy_actions, timesteps, obs_cond)

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss


class StateOnlyTransformerDiffusionPolicy(BasePolicy):
    def __init__(self,
                 state_dim,
                 action_dim,
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
                 num_diffusion_iters):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=pred_horizon,
            n_obs_steps=obs_horizon,
            cond_dim=state_dim,
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
            num_train_timesteps=num_diffusion_iters,
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

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

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