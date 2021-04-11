import numpy as np
import torch 
from torch_utils import misc, persistence
from torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d
from fully_connected_layer import fully_connected_layer as fcl
from modulated_conv2d import modulated_conv2d


class to_rgb_layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()

        self.conv_clamp = conv_clamp
        self.affine = fcl(w_dim, in_channels, bias_init=1)
        
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self):
        styles = self.affine(w) * self.weight_gain

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)

        return x