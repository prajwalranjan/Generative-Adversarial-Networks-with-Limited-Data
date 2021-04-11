# Consists of the Modulated Conv2D layer

import numpy as np
import torch
from torch_utils import misc
from torch_utils.ops import conv2d_resample, fma


def modulated_conv2d(
    x, # input, shape=[batch_size, in_channels, in_height, in_width]
    weight, # weights, shape=[out_channels, in_channels, kernel_height, kernel_width]
    styles, # modulation co-efficients, shape=[batch_size, in_channels]
    noise=None, # to add noise to the output activations
    up=1, # upsampling factpr
    down=1, # downsampling factor
    padding=0, # padding as per upsampled image
    resample_filter=None, 
    demodulate=True, # Weight demodulation
    flip_weight=True,
    fused_modconv=True, # To perform modulation
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw =  weight.shape

    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    misc.assert_shape(styles, [batch_size, in_channels])

    # Normalize inputs
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels*kh*kw) / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)

    # Calculate sample weights and demodultion coefficients
    w = None
    demod_coeff = None

    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)
        w = w + styles.reshape(batch_size, 1, -1, 1, 1)

    if demodulate:
        demod_coeff = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
    
    if demodulate and fused_modconv:
        w = w * demod_coeff.reshape(batch_size, -1, 1, 1, 1)

    # Modulation execution by scaling activations
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight,
        )

        if demodulate and noise is not None:
            x = fma.fma(x, demod_coeff.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * demod_coeff.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        
        return x

    with misc.suppress_tracer_warnings():
        batch_size = int(batch_size)

    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, 1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
        flip_weight=flip_weight,
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])

    if noise is not None:
        x = x.add_(noise)

    return x