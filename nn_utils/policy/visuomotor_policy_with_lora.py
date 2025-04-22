import torch.nn as nn
from nn_utils.policy.visuomotor_policy import VisuomotorUNetDiffusionPolicy, VisuomotorTransformerDiffusionPolicy
from nn_utils.model.lora import LoRAForConv1d, LoRAForConv2d


class VisuomotorUNetDiffusionPolicyWithLoRA(VisuomotorUNetDiffusionPolicy):
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
                 num_inference_diffusion_iters,
                 lora_dropout_p,
                 lora_scale,
                 exclude_critical_layers_from_lora):
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

        self.lora_dropout_p = lora_dropout_p
        self.lora_scale = lora_scale

        self.lora_injected = False

        self.main_params = None
        self.lora_params = None

        #self.exclude_critical_layers_from_lora = exclude_critical_layers_from_lora

        #if self.exclude_critical_layers_from_lora:
        #    self.layers_to_exclude_in_noise_pred_net = self.get_layers_to_exclude_in_noise_pred_net()
        #else:
        #    self.layers_to_exclude_in_noise_pred_net = []

        print("total number of parameters before injecting LoRA:", sum(p.numel() for p in self.parameters()))

    def freeze_main_network(self, apply_lora_to_visual_encoder):
        for param in self.noise_pred_net.parameters():
            param.requires_grad = False

        if apply_lora_to_visual_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if apply_lora_to_visual_encoder:
            for name, param in self.vision_encoder.named_parameters():
                if 'lora_down' in name or 'lora_up' in name:
                    # Keep LoRA parameters trainable
                    param.requires_grad = True
                elif isinstance(param, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for name, param in self.noise_pred_net.named_parameters():
            if 'lora_down' in name or 'lora_up' in name:
                param.requires_grad = True
            elif isinstance(param, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                param.requires_grad = True
            else:
                param.requires_grad = False

        lora_params_list = list(filter(lambda p: p.requires_grad, self.noise_pred_net.parameters())) + list(
            filter(lambda p: p.requires_grad, self.vision_encoder.parameters()))

        total_num_learnable_params = sum(p.numel() for p in lora_params_list)
        # print("Freeze Non-LoRA layers. Now, the number of trainable parameters is:", total_num_learnable_params)

    def get_main_and_lora_params(self):
        return self.main_params, self.lora_params

    def inject_lora(self, apply_lora_to_visual_encoder, lora_rank=16):
        self.noise_pred_net = apply_lora_to_network(self.noise_pred_net, lora_rank, self.lora_dropout_p,
                                                    self.lora_scale, self.layers_to_exclude_in_noise_pred_net)

        if apply_lora_to_visual_encoder:
            self.vision_encoder = apply_lora_to_network(self.vision_encoder, lora_rank, self.lora_dropout_p,
                                                        self.lora_scale, [])

        lora_params = []
        main_params = []

        for name, param in self.noise_pred_net.named_parameters():
            if 'lora_down' in name or 'lora_up' in name:
                lora_params.append(param)
            else:
                main_params.append(param)

        if apply_lora_to_visual_encoder:
            for name, param in self.vision_encoder.named_parameters():
                if 'lora_down' in name or 'lora_up' in name:
                    lora_params.append(param)
                else:
                    main_params.append(param)
        else:
            main_params.extend(self.vision_encoder.parameters())

        self.lora_injected = True

        total_num_lora_params = sum(p.numel() for p in lora_params)


        self.main_params = main_params
        self.lora_params = lora_params

    def merge_lora_weights(self, apply_lora_to_visual_encoder):
        self.noise_pred_net = merge_lora_weights(self.noise_pred_net)

        if apply_lora_to_visual_encoder:
            self.vision_encoder = merge_lora_weights(self.vision_encoder)

        self.lora_injected = False
        #print("LoRA weights have been merged into the main network weights.")



    def reduce_rank(self, new_rank, apply_lora_to_visual_encoder=False):
        self.reduce_lora_rank_in_network(self.noise_pred_net, new_rank)

        if apply_lora_to_visual_encoder:
            self.reduce_lora_rank_in_network(self.vision_encoder, new_rank)

        self.lora_params = []
        for name, param in self.noise_pred_net.named_parameters():
            if 'lora_down' in name or 'lora_up' in name:
                self.lora_params.append(param)

        if apply_lora_to_visual_encoder:
            for name, param in self.vision_encoder.named_parameters():
                if 'lora_down' in name or 'lora_up' in name:
                    self.lora_params.append(param)

    def reduce_lora_rank_in_network(self, network, new_rank):
        for module in network.modules():
            if isinstance(module, LoRAForConv1d) or isinstance(module, LoRAForConv2d):
                module.reduce_rank(new_rank)






class VisuomotorTransformerDiffusionPolicyWithLoRA(VisuomotorTransformerDiffusionPolicy):
    def __init__(self):
        pass





def apply_lora_to_network(network, lora_rank=8, lora_dropout_p=0.1, lora_scale=1.0, layers_to_exclude=[]):
    for name, module in network.named_children():
        module_full_name = f"{network.__class__.__name__}.{name}"
        if module_full_name in layers_to_exclude:
            continue
        if isinstance(module, nn.Conv1d):
            device = module.weight.device
            dtype = module.weight.dtype

            lora_conv = LoRAForConv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                r=lora_rank,
                dropout_p=lora_dropout_p,
                scale=lora_scale,
                device=device,
                dtype=dtype
            )
            lora_conv.conv.weight.data = module.weight.data.clone().to(device=device, dtype=dtype)
            if module.bias is not None:
                lora_conv.conv.bias.data = module.bias.data.clone().to(device=device, dtype=dtype)
            setattr(network, name, lora_conv)
        elif isinstance(module, nn.Conv2d):
            device = module.weight.device
            dtype = module.weight.dtype

            lora_conv = LoRAForConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                r=lora_rank,
                dropout_p=lora_dropout_p,
                scale=lora_scale,
                device=device,
                dtype=dtype
            )
            lora_conv.conv.weight.data = module.weight.data.clone().to(device=device, dtype=dtype)
            if module.bias is not None:
                lora_conv.conv.bias.data = module.bias.data.clone().to(device=device, dtype=dtype)
            setattr(network, name, lora_conv)
        else:
            apply_lora_to_network(module, lora_rank, lora_dropout_p, lora_scale)
    return network


def merge_lora_weights(network):
    for name, module in network.named_children():
        if isinstance(module, LoRAForConv1d) or isinstance(module, LoRAForConv2d):
            module.merge_lora_weights()
            setattr(network, name, module.conv)
        else:
            merge_lora_weights(module)
    return network
