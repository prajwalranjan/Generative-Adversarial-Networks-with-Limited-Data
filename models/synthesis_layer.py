import numpy as np
import torch
from torch_utils import misc, persistence
from torch_utils.ops import bias_act, Conv2dLayer, fma, upfirdn2d
from modulated_conv2d import modulated_conv2d
from fully_connected_layer import fully_connected_layer as fcl


class synthesis_layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, up=1, use_noise=True, activation='lrelu', resample_filter=[1, 3, 3, 1], conv_clamp=None, channels_last= False):
        super().__init__()
        
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = fcl(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))

        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self):
        assert noise_mode in ['random', 'const', 'none']
        
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None

        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        
        return x