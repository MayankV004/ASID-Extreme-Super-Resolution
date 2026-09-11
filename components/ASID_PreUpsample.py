import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.IDSG import IDSG
from ops.IDSG_A import IDSG_A
from ops.pixelshuffle import pixelshuffle_block

class ASID_PreUpsample(nn.Module):
    """
    ASID with Pre-Upsampling Intervention for Extreme Super-Resolution (x8).
    Takes a low-resolution input (H, W), applies a pre-upsampling step (factor 2, bilinear/bicubic),
    and then processes the intermediate features with ASID transformer blocks (upsampling factor 4)
    to achieve the final x8 output (8H, 8W).
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(ASID_PreUpsample, self).__init__()

        res_num = kwargs.get("res_num", 3)
        self.pre_scale = kwargs.get("pre_scale", 2)
        self.post_scale = kwargs.get("post_scale", 4)
        self.pre_mode = kwargs.get("pre_mode", "bilinear")
        bias = kwargs.get("bias", True)

        self.total_scale = self.pre_scale * self.post_scale

        self.res_num = res_num
        self.block0 = IDSG_A(channel_num=num_feat, **kwargs)
        self.block1 = IDSG(channel_num=num_feat, **kwargs)
        self.block2 = IDSG(channel_num=num_feat, **kwargs)

        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, self.post_scale, bias=bias)

        self.window_size = kwargs.get("window_size", 8)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        orig_H, orig_W = x.shape[2:]

        # 1. Pre-upsampling intervention (e.g. 2x bilinear)
        if self.pre_scale > 1:
            x_pre = F.interpolate(x, scale_factor=self.pre_scale, mode=self.pre_mode, align_corners=False if self.pre_mode in ['bilinear', 'bicubic'] else None)
        else:
            x_pre = x

        H_pre, W_pre = x_pre.shape[2:]
        x_pad = self.check_image_size(x_pre)

        # 2. ASID feature extraction and attention-sharing distillation
        residual = self.input(x_pad)
        out, a1, a2, a3, a4, a5, a6 = self.block0(residual)
        out = self.block1(out, a1, a2, a3, a4, a5, a6)
        out = self.block2(out, a1, a2, a3, a4, a5, a6)

        # 3. Residual connection & post-upsampling (e.g. 4x pixel shuffle)
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        # Crop to the exact target 8x output dimensions
        out = out[:, :, :orig_H * self.total_scale, :orig_W * self.total_scale]
        return out
