import numpy as np
import torch
from torch_utils.ops import bias_act


class fully_connected_layer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation='linear', lr_multiplier=1, bias_init=0):
        super().__init__()
        self.activation = activation
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.bias_gain = lr_multiplier
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.weight_gain = lr_multiplier / np.sqrt(in_features)

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias

        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        
        return x