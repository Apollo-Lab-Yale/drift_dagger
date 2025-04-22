import torch
import torch.nn as nn
import torch.nn.functional as F


class RankModulationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U_kept, S_kept, Vh_kept, U_disc, S_disc, Vh_disc,
                stride, padding, dilation, groups, kernel_size, in_channels):
        ctx.input_shape = x.shape
        ctx.save_for_backward(x, U_kept, S_kept, Vh_kept, U_disc, S_disc, Vh_disc)
        ctx.stride = stride if isinstance(stride, int) else stride[0]
        ctx.padding = padding if isinstance(padding, int) else padding[0]
        ctx.dilation = dilation if isinstance(dilation, int) else dilation[0]
        ctx.groups = groups
        ctx.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        ctx.in_channels = in_channels

        Vh_kept_reshaped = Vh_kept.view(-1, in_channels // groups, ctx.kernel_size)
        kept_proj = F.conv1d(x, Vh_kept_reshaped, stride=ctx.stride,
                             padding=ctx.padding, dilation=ctx.dilation, groups=groups)
        kept_scaled = kept_proj * S_kept.view(1, -1, 1)
        U_kept_reshaped = U_kept.unsqueeze(2)
        kept_output = F.conv1d(kept_scaled, U_kept_reshaped)

        ctx.conv1_output_shape = kept_proj.shape
        ctx.conv2_output_shape = kept_output.shape

        if Vh_disc.shape[0] > 0:
            Vh_disc_reshaped = Vh_disc.view(-1, in_channels // groups, ctx.kernel_size)
            disc_proj = F.conv1d(x, Vh_disc_reshaped, stride=ctx.stride,
                                 padding=ctx.padding, dilation=ctx.dilation, groups=groups)
            disc_scaled = disc_proj * S_disc.view(1, -1, 1)
            U_disc_reshaped = U_disc.unsqueeze(2)
            disc_output = F.conv1d(disc_scaled, U_disc_reshaped)
            return kept_output + disc_output

        return kept_output

    @staticmethod
    def backward(ctx, grad_output):
        x, U_kept, S_kept, Vh_kept, U_disc, S_disc, Vh_disc = ctx.saved_tensors
        input_shape = ctx.input_shape

        grad_U_kept = grad_S_kept = grad_Vh_kept = None
        grad_x = None

        if ctx.needs_input_grad[0] or any(ctx.needs_input_grad[1:4]):
            Vh_kept_reshaped = Vh_kept.view(-1, ctx.in_channels // ctx.groups, ctx.kernel_size)

            if ctx.needs_input_grad[1]:  
                Vh_proj = F.conv1d(x, Vh_kept_reshaped, stride=ctx.stride,
                                   padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                scaled_proj = Vh_proj * S_kept.view(1, -1, 1)
                grad_output_reshaped = grad_output.permute(1, 0, 2).reshape(grad_output.size(1), -1)
                scaled_proj_reshaped = scaled_proj.permute(0, 2, 1).reshape(-1, S_kept.size(0))
                grad_U_kept = grad_output_reshaped @ scaled_proj_reshaped

            if ctx.needs_input_grad[2]:  
                Vh_proj = F.conv1d(x, Vh_kept_reshaped, stride=ctx.stride,
                                   padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
                grad_output_reshaped = grad_output.permute(0, 2, 1).reshape(-1, grad_output.size(1))
                Vh_proj_reshaped = Vh_proj.permute(0, 2, 1).reshape(-1, Vh_proj.size(1))
                grad_S_kept = (U_kept.t() @ grad_output_reshaped.t() @ Vh_proj_reshaped).diag()

            if ctx.needs_input_grad[3]: 
                U_kept_reshaped = U_kept.unsqueeze(2)
                grad_output_proj = F.conv1d(grad_output, U_kept_reshaped.transpose(0, 1))
                grad_output_scaled = grad_output_proj * S_kept.view(1, -1, 1)

                batch_size = x.size(0)
                in_channels = ctx.in_channels
                kernel_size = ctx.kernel_size
                r = S_kept.size(0)

                grad_output_scaled = grad_output_scaled.transpose(1, 2).reshape(-1, r)

                if ctx.padding > 0:
                    x_padded = F.pad(x, (ctx.padding, ctx.padding))
                else:
                    x_padded = x

                L_out = grad_output_scaled.size(0) // batch_size
                L_in = x_padded.size(2) - kernel_size + 1

                x_strided = x_padded.unfold(2, kernel_size, ctx.stride)
                x_strided = x_strided.permute(0, 2, 1, 3).contiguous()
                x_strided = x_strided.view(-1, in_channels * kernel_size)

                grad_Vh_kept = grad_output_scaled.t() @ x_strided

            if ctx.needs_input_grad[0]:  
                total_output_size = input_shape[2]
                expected_output_size = (grad_output.size(2) - 1) * ctx.stride - 2 * ctx.padding + ctx.kernel_size
                output_padding = total_output_size - expected_output_size

                U_kept_reshaped = U_kept.unsqueeze(2)
                grad_output_proj = F.conv1d(grad_output, U_kept_reshaped.transpose(0, 1))
                grad_output_scaled = grad_output_proj * S_kept.view(1, -1, 1)

                grad_x = F.conv_transpose1d(
                    grad_output_scaled, Vh_kept_reshaped,
                    stride=ctx.stride,
                    padding=ctx.padding,
                    output_padding=output_padding if output_padding > 0 else 0,
                    groups=ctx.groups,
                    dilation=ctx.dilation
                )

                if Vh_disc.shape[0] > 0:
                    Vh_disc_reshaped = Vh_disc.view(-1, ctx.in_channels // ctx.groups, ctx.kernel_size)
                    grad_output_proj_disc = F.conv1d(grad_output, U_disc.unsqueeze(2).transpose(0, 1))
                    grad_output_scaled_disc = grad_output_proj_disc * S_disc.view(1, -1, 1)
                    grad_x_disc = F.conv_transpose1d(
                        grad_output_scaled_disc, Vh_disc_reshaped,
                        stride=ctx.stride,
                        padding=ctx.padding,
                        output_padding=output_padding if output_padding > 0 else 0,
                        groups=ctx.groups,
                        dilation=ctx.dilation
                    )
                    grad_x = grad_x + grad_x_disc

        return (grad_x, grad_U_kept, grad_S_kept, grad_Vh_kept,
                None, None, None, None, None, None, None, None, None)


class RankModulationForConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, original_module, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, r=4, device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.device = device
        self.dtype = dtype

        weight = original_module.weight.data.view(self.out_channels, -1)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        self.trainable_rank = min(r, S.shape[0])

        self.U_kept = nn.Parameter(U[:, :self.trainable_rank].clone().detach())
        self.S_kept = nn.Parameter(S[:self.trainable_rank].clone().detach())
        self.Vh_kept = nn.Parameter(Vh[:self.trainable_rank, :].clone().detach())

        if self.trainable_rank < S.shape[0]:
            U_disc = U[:, self.trainable_rank:].clone().detach()
            S_disc = S[self.trainable_rank:].clone().detach()
            Vh_disc = Vh[self.trainable_rank:, :].clone().detach()
        else:
            U_disc = torch.empty((U.shape[0], 0), device=U.device, dtype=U.dtype)
            S_disc = torch.empty(0, device=S.device, dtype=S.dtype)
            Vh_disc = torch.empty((0, Vh.shape[1]), device=Vh.device, dtype=Vh.dtype)

        self.register_buffer('U_disc', U_disc)
        self.register_buffer('S_disc', S_disc)
        self.register_buffer('Vh_disc', Vh_disc)

    def forward(self, x):
        U_kept = self.U_kept.detach().requires_grad_(True)
        S_kept = self.S_kept.detach().requires_grad_(True)
        Vh_kept = self.Vh_kept.detach().requires_grad_(True)
        U_disc = self.U_disc.detach()
        S_disc = self.S_disc.detach()
        Vh_disc = self.Vh_disc.detach()

        output = RankModulationFunction.apply(
            x, U_kept, S_kept, Vh_kept,
            U_disc, S_disc, Vh_disc,
            self.stride, self.padding, self.dilation, self.groups,
            self.kernel_size[0], self.in_channels
        )

        if self.training:
            if U_kept.grad is not None:
                self.U_kept.grad = U_kept.grad
            if S_kept.grad is not None:
                self.S_kept.grad = S_kept.grad
            if Vh_kept.grad is not None:
                self.Vh_kept.grad = Vh_kept.grad

        return output

    def reduce_trainable_rank(self, new_trainable_rank):
        if new_trainable_rank >= self.trainable_rank:
            return  

        with torch.no_grad():
            U_current = self.U_kept * self.S_kept.view(1, -1)
            Vh_current = self.Vh_kept

            U, S, Vh = torch.linalg.svd(U_current @ Vh_current, full_matrices=False)

            self.U_kept = nn.Parameter(U[:, :new_trainable_rank].clone().detach())
            self.S_kept = nn.Parameter(S[:new_trainable_rank].clone().detach())
            self.Vh_kept = nn.Parameter(Vh[:new_trainable_rank, :].clone().detach())

            if new_trainable_rank < S.shape[0]:
                U_disc_new = U[:, new_trainable_rank:].clone().detach()
                S_disc_new = S[new_trainable_rank:].clone().detach()
                Vh_disc_new = Vh[new_trainable_rank:, :].clone().detach()

                if self.U_disc.shape[1] > 0:
                    self.U_disc = torch.cat([self.U_disc, U_disc_new], dim=1)
                    self.S_disc = torch.cat([self.S_disc, S_disc_new])
                    self.Vh_disc = torch.cat([self.Vh_disc, Vh_disc_new], dim=0)
                else:
                    self.U_disc = U_disc_new
                    self.S_disc = S_disc_new
                    self.Vh_disc = Vh_disc_new

        self.trainable_rank = new_trainable_rank