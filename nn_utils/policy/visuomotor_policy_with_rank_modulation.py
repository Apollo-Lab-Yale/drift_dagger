import torch.nn as nn
import torch
from nn_utils.policy.visuomotor_policy import VisuomotorUNetDiffusionPolicy, VisuomotorTransformerDiffusionPolicy
from nn_utils.model.lora import LoRAForConv1d, LoRAForConv2d
from nn_utils.model.rank_modulation import RankModulationForConv1d
from nn_utils.model.unet import ConditionalUnet1D, ConditionalResidualBlock1D, Downsample1d, Upsample1d, Conv1dBlock


class VisuomotorUNetDiffusionPolicyWithRankModulation(VisuomotorUNetDiffusionPolicy):
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
        super().__init__(state_dim,
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
                         num_inference_diffusion_iters)



        self.inject_rank_modulation(initial_rank=self.max_rank)

        self.scaler = torch.cuda.amp.GradScaler()
 
     def compute_loss(self, batch):
         with torch.cuda.amp.autocast():
             loss = super().compute_loss(batch)
 
             return loss
 
     def predict_action(self, obs):
         with torch.cuda.amp.autocast():
             action = super().predict_action(obs)
 
             return action

    def inject_rank_modulation(self, initial_rank=16):
        self.noise_pred_net = apply_rank_modulation_to_network(
            self.noise_pred_net, initial_rank, self.layers_to_exclude_in_noise_pred_net
        )

        rank_mod_params = []
        main_params = []

        for name, param in self.noise_pred_net.named_parameters():
            if 'trainable_down' in name or 'trainable_up' in name:
                rank_mod_params.append(param)
            else:
                main_params.append(param)

        main_params.extend(self.vision_encoder.parameters())

        total_num_rank_mod_params = sum(p.numel() for p in rank_mod_params)

    def reduce_rank(self, new_rank):
        self.reduce_trainable_rank_in_network(self.noise_pred_net, new_rank)

    def reduce_trainable_rank_in_network(self, network, new_rank):
        for module in network.modules():
            if isinstance(module, RankModulationForConv1d):
                module.reduce_trainable_rank(new_rank)


def apply_rank_modulation_to_network(network, initial_rank=8, layers_to_exclude=[]):
    for name, module in network.named_children():
        module_full_name = f"{network.__class__.__name__}.{name}"
        if module_full_name in layers_to_exclude:
            continue
        if isinstance(module, nn.Conv1d):
            device = module.weight.device
            dtype = module.weight.dtype

            rank_mod_conv = RankModulationForConv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                original_module=module,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                r=initial_rank,
                device=device,
                dtype=dtype
            )

            setattr(network, name, rank_mod_conv)
        else:
            apply_rank_modulation_to_network(module, initial_rank, layers_to_exclude)

    return network
