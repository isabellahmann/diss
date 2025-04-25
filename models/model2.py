# Adapted code from Paula Seidler

import sys
sys.path.append('../')

import torch
import torch.nn as nn
from enum import Enum

from models.supn_blocks import ResnetBlock, ResNetType
from supn_base.sparse_precision_cholesky import get_num_off_diag_weights
from supn_base.supn_data import SUPNData

class SupnDecoderType(Enum):
    DIRECT = 3

class ResNetUnetSingleDecoder(nn.Module):
    def __init__(self, in_channels=1, num_init_features=32, use_skip_connections=True,
                 max_dropout=0.0, use_bias=True, use_3d=False, max_ch=512):
        super().__init__()
        conv_fn = nn.Conv3d if use_3d else nn.Conv2d
        features = num_init_features
        self.use_skip = use_skip_connections
        self.max_dropout = max_dropout

        self._conv1 = conv_fn(in_channels, features, kernel_size=3, padding=1, bias=use_bias)

        self.encoder = ResnetBlock(features, min(max_ch, features * 2),
                                   block_type=ResNetType.DOWNSAMPLE, use_bnorm=True,
                                   use_bias=use_bias, use_3D=use_3d)

        self.bottleneck = ResnetBlock(min(max_ch, features * 2), min(max_ch, features * 2),
                                      block_type=ResNetType.SAME, use_bnorm=True,
                                      use_bias=use_bias, use_3D=use_3d)

        features_dec_in = min(max_ch * 2, features * 2) * 2 if use_skip_connections else min(max_ch, features * 2)
        self.mean_decoder = ResnetBlock(features_dec_in, features, block_type=ResNetType.UPSAMPLE,
                                        use_bnorm=True, use_bias=use_bias, use_3D=use_3d)
        self.supn_decoder = ResnetBlock(features_dec_in, features, block_type=ResNetType.UPSAMPLE,
                                        use_bnorm=True, use_bias=use_bias, use_3D=use_3d)

    def forward(self, x):
        enc = self._conv1(x)
        enc = self.encoder(enc)
        enc = self.bottleneck(enc)
        skip = enc

        if self.use_skip:
            x = torch.cat((enc, skip), dim=1)
            mean = self.mean_decoder(torch.cat((enc, skip), dim=1))
            supn = self.supn_decoder(x)
        else:
            mean = self.mean_decoder(enc)
            supn = self.supn_decoder(enc)

        return supn, {'mean': mean, 'supn': supn}


class SUPNDecoderDirect(nn.Module):
    def __init__(self, in_channels, num_log_diags, num_off_diags, use_bias=True):
        super().__init__()
        conv = nn.Conv2d
        self.pre_conv = conv(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=use_bias)
        self.log_diag_conv = conv(in_channels, num_log_diags, kernel_size=3, padding=1, padding_mode='reflect', bias=use_bias)
        self.off_diag_conv = conv(in_channels, num_off_diags, kernel_size=3, padding=1, padding_mode='reflect', bias=use_bias)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.pre_conv(x))
        log_diag = self.log_diag_conv(x)
        off_diag = self.off_diag_conv(x)
        return log_diag, off_diag, None


class MeanDecoder(nn.Module):
    def __init__(self, in_channels, num_log_diags, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_log_diags, kernel_size=3, padding=1, padding_mode='reflect', bias=use_bias)

    def forward(self, x):
        return self.conv(x)


class SUPNEncoderDecoder(nn.Module):
    def __init__(self, in_channels, num_out_ch, num_init_features, num_log_diags, local_connection_dist,
                 use_skip_connections=True, max_dropout=0.0, use_bias=False, use_3D=False, max_ch=512):
        super().__init__()

        self.local_connection_dist = local_connection_dist
        self.use_3d = use_3D
        num_off_diags = get_num_off_diag_weights(local_connection_dist, use_3D) * in_channels

        self.encoder_decoder = ResNetUnetSingleDecoder(
            in_channels, num_init_features, use_skip_connections, max_dropout, use_bias, use_3D, max_ch
        )

        feature_dim = min(max_ch, num_init_features * 2)
        self.supn_decoder = SUPNDecoderDirect(feature_dim, num_log_diags, num_off_diags, use_bias)
        self.mean_decoder = MeanDecoder(feature_dim, num_log_diags, use_bias)

    def forward(self, x):
        final_feat, dec = self.encoder_decoder(x)
        mean = self.mean_decoder(dec['mean'])
        log_diag, off_diag, cross_ch = self.supn_decoder(dec['supn'])

        return [torch.distributions.SUPN(SUPNData(
            mean=mean, log_diag=log_diag, off_diag=off_diag, cross_ch=cross_ch,
            local_connection_dist=self.local_connection_dist
        ))]
