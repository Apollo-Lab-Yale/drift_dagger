import torch
import torch.nn as nn

class LoRAForConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, r=4, dropout_p=0.1, scale=1.0,
                 device=None, dtype=None):
        super().__init__()
        max_rank = min(in_channels, out_channels)
        #print("Max rank for LoRA: ", max_rank)
        if r > min(in_channels, out_channels):
            #print(f"LoRA rank {r} must be less than or equal to {max_rank}")
            #print("Override rank to max rank")
            r = max_rank
        self.r = r
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.lora_down = nn.Conv1d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_up = nn.Conv1d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device,
            dtype=dtype,
        )

        #self.dropout = nn.Dropout(dropout_p)
        self.selector = nn.Identity()

        #nn.init.normal_(self.lora_down.weight, std=1 / r)
        #nn.init.zeros_(self.lora_up.weight)
        #nn.init.normal_(self.lora_down.weight, mean=0.0, std=0.01)
        #nn.init.normal_(self.lora_up.weight, mean=0.0, std=0.01)
        gain = 1.0
        nn.init.xavier_normal_(self.lora_down.weight, gain=gain)
        nn.init.xavier_normal_(self.lora_up.weight, gain=gain)

    def forward(self, x):
        conv_output = self.conv(x)
        lora_output = self.lora_up(self.selector(self.lora_down(x)))
        #lora_output = self.dropout(lora_output) * self.scale
        lora_output = lora_output * self.scale
        return conv_output + lora_output

    def merge_lora_weights(self):
        lora_up_weight = self.lora_up.weight.view(self.out_channels, self.r)
        lora_down_weight = self.lora_down.weight.view(self.r, -1)  

        effective_weight = lora_up_weight @ lora_down_weight  
        effective_weight = effective_weight.view(self.conv.weight.shape)  

        self.conv.weight.data += effective_weight * self.scale

        del self.lora_down
        del self.lora_up
        #del self.dropout
        del self.selector

        self.forward = self.conv.forward

    def reduce_rank(self, new_rank):
        if new_rank >= self.r:
            return

        device = self.lora_up.weight.device
        dtype = self.lora_up.weight.dtype

        lora_up_weight = self.lora_up.weight.data.view(self.out_channels, self.r)
        lora_down_weight = self.lora_down.weight.data.view(self.r, -1)  

        delta_W = lora_up_weight @ lora_down_weight  

        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        U = U.to(device=device, dtype=dtype)
        S = S.to(device=device, dtype=dtype)
        Vh = Vh.to(device=device, dtype=dtype)

        U_kept = U[:, :new_rank]
        S_kept = S[:new_rank]
        Vh_kept = Vh[:new_rank, :]

        U_discarded = U[:, new_rank:]
        S_discarded = S[new_rank:]
        Vh_discarded = Vh[new_rank:, :]

        delta_W_discarded = (U_discarded * S_discarded.unsqueeze(0)) @ Vh_discarded

        self.conv.weight.data += delta_W_discarded.view_as(self.conv.weight) * self.scale

        sqrt_S_kept = torch.sqrt(S_kept)
        lora_up_weight_new = U_kept * sqrt_S_kept.unsqueeze(0)
        lora_down_weight_new = sqrt_S_kept.unsqueeze(1) * Vh_kept

        lora_up_weight_new = lora_up_weight_new.view(self.out_channels, new_rank, 1)
        lora_down_weight_new = lora_down_weight_new.view(new_rank, self.in_channels, self.kernel_size[0])

        self.r = new_rank
        self.lora_up = nn.Conv1d(
            in_channels=new_rank,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.lora_down = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=new_rank,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.conv.groups,
            bias=False,
            device=device,
            dtype=dtype
        )

        self.lora_up.weight.data = lora_up_weight_new
        self.lora_down.weight.data = lora_down_weight_new

class LoRAForConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, r=4, dropout_p=0.1, scale=1.0,
                 device=None, dtype=None):
        super().__init__()
        max_rank = min(in_channels, out_channels)
        #print("Max rank for LoRA: ", max_rank)
        if r > min(in_channels, out_channels):
            print(f"LoRA rank {r} must be less than or equal to {max_rank}")
            print("Override rank to max rank")
            r = max_rank
        self.r = r
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.dropout = nn.Dropout(dropout_p)
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        conv_output = self.conv(x)
        lora_output = self.lora_up(self.selector(self.lora_down(x)))
        lora_output = self.dropout(lora_output) * self.scale
        return conv_output + lora_output

    def merge_lora_weights(self):
        lora_up_weight = self.lora_up.weight.view(self.out_channels, self.r)
        lora_down_weight = self.lora_down.weight.view(self.r, -1)  # [r, in_channels * kh * kw]

        effective_weight = lora_up_weight @ lora_down_weight  # [out_channels, in_channels * kh * kw]
        effective_weight = effective_weight.view(self.conv.weight.shape)  # [out_channels, in_channels, kh, kw]

        self.conv.weight.data += effective_weight * self.scale

        del self.lora_down
        del self.lora_up
        del self.dropout
        del self.selector

        self.forward = self.conv.forward

    def reduce_rank(self, new_rank):
        if new_rank >= self.r:
            return

        device = self.lora_up.weight.device
        dtype = self.lora_up.weight.dtype

        lora_up_weight = self.lora_up.weight.data.view(self.out_channels, self.r)
        lora_down_weight = self.lora_down.weight.data.view(self.r,
                                                           -1)  

        delta_W = lora_up_weight @ lora_down_weight 

        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        U = U.to(device=device, dtype=dtype)
        S = S.to(device=device, dtype=dtype)
        Vh = Vh.to(device=device, dtype=dtype)

        U_kept = U[:, :new_rank]
        S_kept = S[:new_rank]
        Vh_kept = Vh[:new_rank, :]

        U_discarded = U[:, new_rank:]
        S_discarded = S[new_rank:]
        Vh_discarded = Vh[new_rank:, :]

        delta_W_discarded = (U_discarded * S_discarded.unsqueeze(0)) @ Vh_discarded

        self.conv.weight.data += delta_W_discarded.view_as(self.conv.weight) * self.scale

        sqrt_S_kept = torch.sqrt(S_kept)
        lora_up_weight_new = U_kept * sqrt_S_kept.unsqueeze(0)
        lora_down_weight_new = sqrt_S_kept.unsqueeze(1) * Vh_kept

        lora_up_weight_new = lora_up_weight_new.view(self.out_channels, new_rank, 1, 1)
        lora_down_weight_new = lora_down_weight_new.view(
            new_rank, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )

        self.r = new_rank
        self.lora_up = nn.Conv2d(
            in_channels=new_rank,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.lora_down = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=new_rank,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.conv.groups,
            bias=False,
            device=device,
            dtype=dtype
        )

        self.lora_up.weight.data = lora_up_weight_new
        self.lora_down.weight.data = lora_down_weight_new
