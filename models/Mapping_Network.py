import numpy as np
import torch
from torch_utils import misc, persistence
from torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d
from fully_connected_layer import fully_connected_layer as fcl


class mapping_network(torch.nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, num_ws, num_layers=8, embed_features=None, layer_features=None, activation='lrelu', lr_multiplier=0.01, w_avg_beta=0.995):
        super().__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        
        if c_dim == 0:
            embed_features = 0
        
        if layer_features is None:
            layer_features = w_dim
        
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = fcl(c_dim, embed_features)

        for index in range(num_layers):
            in_features = features_list[index]
            out_features = features_list[index + 1]
            layer = fcl(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{index}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        x = None

        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        for index in range(self.num_layers):
            layer = getattr(self, f'fc{index}')
            x = layer(x)

        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None

                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x