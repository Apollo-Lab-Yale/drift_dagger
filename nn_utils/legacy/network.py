#####################################
#@markdown ### **Network**
from nn_utils.model.vision_encoder import get_resnet, replace_bn_with_gn
from nn_utils.model.unet import ConditionalUnet1D, Conv1dBlock
from nn_utils.model.lora import LoRAForConv1d
from torch import nn

"""
def apply_lora_to_network(network, lora_rank=8, lora_dropout_p=0.1, lora_scale=1.0):
    for name, module in network.named_children():
        if isinstance(module, Conv1dBlock):
            # Replace Conv1dBlock with LORA_Conv1dBlock
            lora_conv_block = Conv1dLoRABlock(
                inp_channels=module.block[0].in_channels,
                out_channels=module.block[0].out_channels,
                kernel_size=module.block[0].kernel_size[0],
                n_groups=module.block[1].num_groups,
                r=lora_rank,
                dropout_p=lora_dropout_p,
                scale=lora_scale
            )
            # Copy the weights from the original Conv1d to the LoRA Conv1d
            lora_conv_block.block[0].conv.weight.data = module.block[0].weight.data.clone()
            if module.block[0].bias is not None:
                lora_conv_block.block[0].conv.bias.data = module.block[0].bias.data.clone()

            # Replace the Conv1dBlock in the network with the LoRA Conv1dBlock
            setattr(network, name, lora_conv_block)

        elif isinstance(module, nn.Conv1d):
            # Replace Conv1d with LoraInjectedConv1d
            lora_conv = LoRAForConv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0],
                dilation=module.dilation[0],
                groups=module.groups,
                bias=(module.bias is not None),
                r=lora_rank,
                dropout_p=lora_dropout_p,
                scale=lora_scale
            )
            # Copy the weights from the original Conv1d to the LoRA Conv1d
            lora_conv.conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_conv.conv.bias.data = module.bias.data.clone()

            # Replace the Conv1d in the network with the LoRA Conv1d
            setattr(network, name, lora_conv)

        else:
            # Recursively apply to child modules
            apply_lora_to_network(module, lora_rank, lora_dropout_p, lora_scale)

    return network
"""


def create_nets(vision_feature_dim, lowdim_obs_dim, action_dim, obs_horizon):
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # observation feature has 514 dims in total per step

    obs_dim = vision_feature_dim + lowdim_obs_dim
    # Print OBS_DIM to check the total observation dimension
    print(f"OBS_DIM: {obs_dim}")
    # create network object
    print(f"GLOBALCONDDIM: ", obs_dim*obs_horizon)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    return nets

"""
def create_LORA_nets(vision_feature_dim, lowdim_obs_dim, action_dim, obs_horizon, original_nets):
    from copy import deepcopy

    # Copy the original networks
    vision_encoder = deepcopy(original_nets['vision_encoder'])
    noise_pred_net = deepcopy(original_nets['noise_pred_net'])
    for param in noise_pred_net.parameters():
        param.requires_grad = False
    for param in vision_encoder.parameters():
        param.requires_grad = True
    # vision_encoder = apply_lora_to_network(vision_encoder)
    noise_pred_net = apply_lora_to_network(noise_pred_net)

    # The final architecture has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    print("LORA nets created")
    return nets
"""

# def create_LORA_nets(vision_feature_dim, lowdim_obs_dim, action_dim, obs_horizon):
#     # Construct ResNet18 encoder
#     # If you have multiple camera views, use separate encoder weights for each view.
#     vision_encoder = get_resnet('resnet18')
#
#     # IMPORTANT!
#     # Replace all BatchNorm with GroupNorm to work with EMA
#     # Performance will tank if you forget to do this!
#     vision_encoder = replace_bn_with_gn(vision_encoder)
#
#     # Observation feature has (vision_feature_dim + lowdim_obs_dim) dims in total per step
#     obs_dim = vision_feature_dim + lowdim_obs_dim
#     print(f"OBS_DIM: {obs_dim}")
#
#     # Create noise prediction network with LoRA
#     print(f"GLOBALCONDDIM: {obs_dim * obs_horizon}")
#     noise_pred_net = LORAConditionalUnet1D(
#         input_dim=action_dim,
#         global_cond_dim=obs_dim * obs_horizon
#     )
#
#     # The final architecture has 2 parts
#     nets = nn.ModuleDict({
#         'vision_encoder': vision_encoder,
#         'noise_pred_net': noise_pred_net
#     })
#
#     return nets


def analyze_parameter_distribution(model):
    layer_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            layer_params[name] = params
    # Sort layers by parameter count in descending order
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_layers:
        print(f"{name}: {count} parameters")
    total_params = sum(layer_params.values())
    print(f"Total analyzed parameters: {total_params}")


def verify_freezing(model):
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print("\n===== Trainable Parameters =====")
    for name in trainable_params:
        print(f"  {name}")
    print("\n===== Frozen Parameters =====")
    for name in frozen_params:
        print(f"  {name}")
    print(f"\nTotal Trainable Parameters: {len(trainable_params)}")
    print(f"Total Frozen Parameters: {len(frozen_params)}\n")


