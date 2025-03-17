# Adapted code from Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk),
# with the following key changes and adjustments:
# -> The model now infers SUPN (Sparse Uncertainty Prediction Network) objects at multiple intermediate layers, 
#    not just the last layer, with adaptive control via the `nr_of_prediction_scales`.
# -> Outputs the SUPN objects as a list of layers, each with corresponding scalings.
# -> For each SUPN object, the `nr_of_connections` (i.e., local connection distances) is provided, 
#    and it must match the length of the scaling levels (`nr_of_scaling`).

## currently adapted from Paula's code

import sys
sys.path.append('../')

import torch
import torch.nn as nn
from enum import Enum

from models.supn_blocks import ResnetBlock, ResNetType

from supn_base.sparse_precision_cholesky import get_num_off_diag_weights
from supn_base.cholespy_solver import get_num_off_diag_weights
from supn_base.supn_data import SUPNData
from supn_base.supn_distribution import SUPN


class SupnDecoderType(Enum):
    ORTHONORMAL_BASIS = 0
    BASIS = 2
    DIRECT = 3
    SOFTMAX_BASIS = 4

class ResNetUnet2Decoders(nn.Module):
    def __init__(self, in_channels=1, num_init_features=32, use_skip_connections=True, num_scales=4, max_dropout=0.0, use_bias=True, 
                 use_3d=False, max_ch=512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_batch_norm = True
        features = num_init_features

        self._use_batch_norm = use_batch_norm
        self._use_skip_connections = use_skip_connections
        self.max_dropout = max_dropout
        
        self.encoders = []
        self.do_layers = []
        self.mean_decoder = []
        self.supn_decoder = []
        conv_fn = nn.Conv3d if use_3d else nn.Conv2d
        self._conv1 = conv_fn(in_channels, features, kernel_size=3, padding=1, bias=use_bias)


        # Encoder layers
        for level in range(1, num_scales +1 ):
            block = ResnetBlock(in_channels=min(max_ch, features * pow(2, level - 1)),
                                out_channels=min(max_ch, features * pow(2, level)),
                                block_type=ResNetType.DOWNSAMPLE,
                                use_bnorm=use_batch_norm,
                                use_bias=use_bias,
                                use_3D=use_3d)
            
            if self.max_dropout > 0.0:
                self.do_layers.append(torch.nn.Dropout(p=max_dropout * (level / (num_scales))))
            self.encoders.append(block)

        #bottleneck
        self.bottleneck = ResnetBlock(in_channels=min(max_ch, features * int(pow(2, num_scales + 1))),
                                      out_channels=min(max_ch, features * int(pow(2, num_scales + 1))),
                                      block_type=ResNetType.SAME,
                                      use_bnorm=use_batch_norm,
                                      use_bias=use_bias,
                                      use_3D=use_3d)
        
        #Decoder Layers
        for level in range(num_scales, 0, -1):
            features_dec_in = min(max_ch*2, features * pow(2, level)) *2 if self._use_skip_connections else min(max_ch, features * pow(2, level))
            mean_block = ResnetBlock(in_channels=features_dec_in,
                                out_channels=min(max_ch, features * pow(2, level - 1)),
                                block_type=ResNetType.UPSAMPLE,
                                use_bnorm=use_batch_norm,
                                use_bias=use_bias,
                                use_3D=use_3d)
            supn_block= ResnetBlock(in_channels=features_dec_in,
                                out_channels=min(max_ch, features * pow(2, level - 1)),
                                block_type=ResNetType.UPSAMPLE,
                                use_bnorm=use_batch_norm,
                                use_bias=use_bias,
                                use_3D=use_3d)
            
            self.mean_decoder.append(mean_block)
            self.supn_decoder.append(supn_block)
        
        # Final decoder blocks
        final_in = min(max_ch, features * 2 if use_skip_connections else features)
        final_out = min(max_ch, num_init_features)
        self.mean_decoder.append(ResnetBlock(in_channels=final_in,
                                out_channels=final_out,
                                block_type=ResNetType.UPSAMPLE,
                                use_bnorm=use_batch_norm,
                                use_bias=use_bias,
                                use_3D=use_3d))
        self.supn_decoder.append(ResnetBlock(in_channels=final_in,
                                out_channels=final_out,
                                block_type=ResNetType.UPSAMPLE,
                                use_bnorm=use_batch_norm,
                                use_bias=use_bias,
                                use_3D=use_3d))
        self.encoders = torch.nn.ModuleList(self.encoders)
        self.mean_decoder = torch.nn.ModuleList(self.mean_decoder)
        self.supn_decoder = torch.nn.ModuleList(self.supn_decoder)
        self.do_layers = torch.nn.ModuleList(self.do_layers)

    def forward(self, x):
        '''
        Forward loop -> input data(image) -> to some intermediate representations in the 2 decoders before transforming them into mean and supn objects
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            x is the final encoding
            dicts of encodings and decodings at ...
        """
        '''
        encodings = []
        decodings_mean = []
        decodings_supn = []

        # Initial convolution layer
        x = self._conv1(x)

        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            if len(self.do_layers) > 0:
                x = self.do_layers[idx](x)
            encodings.append(x)

        x = self.bottleneck(x)
        if len(self.do_layers) > 0:
            x = self.do_layers[-1](x)

        encodings.reverse()
        mean = x
        for idx, encoding in enumerate(encodings):
            if self._use_skip_connections:
                x = torch.cat((x, mean), dim=1)
                mean = torch.cat((mean, encoding), dim=1)
            mean = self.mean_decoder[idx](mean)
            x = self.supn_decoder[idx](x)

            decodings_mean.append(mean)
            decodings_supn.append(x)

        return x, {'encodings': encodings, 'decodings_mean': decodings_mean, 'decodings_supn': decodings_supn}



class SUPNDecoderBase(nn.Module):
    def __init__(self, num_init_features, num_log_diags, num_off_diags, decoder_type: SupnDecoderType,
                 num_cross_ch_images=None, predict_log_diags=True, dropout_rate=0.0, smooth_diags=True, smooth_off_diags=True,
                 supn_res_level: int = 1):
        super().__init__()

        self.num_log_diags = num_log_diags
        self.num_off_diags = num_off_diags
        self.num_cross_ch_images = num_cross_ch_images
        self.decoder_type = decoder_type

        assert supn_res_level >= 1
        self.supn_res_level = supn_res_level

        if supn_res_level > 2:
            num_init_features = num_init_features * pow(2, supn_res_level - 2)

        self.num_init_features = num_init_features

        self.predict_log_diags = predict_log_diags
        self.do_layer = torch.nn.Dropout(p=dropout_rate)
        self.smooth_diags = smooth_diags
        self.smooth_off_diags = smooth_off_diags

        if self.smooth_diags:
            self.smoothing_kernel = torch.nn.Parameter(torch.randn(1, 1, 5, 5))
        if self.smooth_off_diags:
            self.off_smoothing_kernel = torch.nn.Parameter(torch.randn(1, 1, 5, 5))

    def apply_smoothing(self, log_diags, off_diags):
        if self.smooth_diags:
            kernel = torch.nn.functional.sigmoid(self.smoothing_kernel)
            kernel = kernel / (torch.sum(kernel)+1e-5)
            kernel = torch.cat([kernel for i in range(log_diags.shape[1])], 0)
            diag = torch.nn.functional.pad(torch.exp(log_diags), (2, 2, 2, 2), mode='reflect')
            log_diags = torch.log(torch.nn.functional.conv2d(diag, kernel, groups=log_diags.shape[1]))

        if self.smooth_off_diags:
            kernel = torch.nn.functional.sigmoid(self.off_smoothing_kernel)
            kernel = kernel / (torch.sum(kernel)+1e-5)
            off_smooth = torch.cat([kernel for i in range(off_diags.shape[1])], 0)
            off_diags = torch.nn.functional.conv2d(torch.nn.functional.pad(off_diags, (2, 2, 2, 2), mode='reflect'),
                                                   off_smooth, groups=off_diags.shape[1])

        return log_diags, off_diags

    def get_encoding(self, decodings):
        decoding = decodings[- self.supn_res_level]
        return decoding



class SUPNDecoderDirect(SUPNDecoderBase):
    def __init__(self, num_init_features, num_log_diags, num_off_diags, num_cross_ch_images=None, use_bias=True, num_scales = 4, use_3D=False, max_ch=512, **kwargs):
        """
        Decoder for the SUPN model, dynamically handling predictions for multiple levels.
        """
        super().__init__(num_init_features, num_log_diags, num_off_diags, decoder_type=SupnDecoderType.DIRECT, 
                         num_cross_ch_images=num_cross_ch_images, **kwargs)

        conv_class = nn.Conv3d if use_3D else nn.Conv2d
        

        # Per-layer convolutional prediction blocks
        self.pre_convs = nn.ModuleList()

        self.log_diag_convs = nn.ModuleList()
        self.off_diag_convs = nn.ModuleList()
        self.cross_ch_convs = nn.ModuleList() if num_cross_ch_images is not None else None

        # Dynamically initialize layers for each decoder level
        features = min(max_ch, num_init_features * pow(2, 0))

        # Pre-conv layer for processing input
        self.pre_convs.append(conv_class(features, features, kernel_size=3, padding=1,
                                            padding_mode='reflect', bias=use_bias)) # need quite a few features -> accounting for off_diags, and log_diag -> adjust!

        # Prediction layers
        self.log_diag_convs.append(conv_class(features, num_log_diags, kernel_size=3, padding=1,
                                                padding_mode='reflect', bias=use_bias))
        self.off_diag_convs.append(conv_class(features, num_off_diags[0], kernel_size=3, padding=1,
                                      padding_mode='reflect', bias=use_bias))
        
        if num_cross_ch_images is not None:
            self.cross_ch_convs.append(conv_class(features, num_cross_ch_images, kernel_size=3, padding=1,
                                                    padding_mode='reflect', bias=use_bias))
        for level in range(num_scales-1):  # Loop backward (from largest features)
            features = min(max_ch, num_init_features * pow(2, level+1))

            # Pre-conv layer for processing input
            self.pre_convs.append(conv_class(features, features, kernel_size=3, padding=1,
                                             padding_mode='reflect', bias=use_bias))

            # Prediction layers
            self.log_diag_convs.append(conv_class(features, num_log_diags, kernel_size=3, padding=1,
                                                  padding_mode='reflect', bias=use_bias))
            self.off_diag_convs.append(conv_class(features, num_off_diags[level+1], kernel_size=3, padding=1,
                                                  padding_mode='reflect', bias=use_bias))

            if num_cross_ch_images is not None:
                self.cross_ch_convs.append(conv_class(features, num_cross_ch_images, kernel_size=3, padding=1,
                                                      padding_mode='reflect', bias=use_bias))

        #print(self.pre_convs,self.log_diag_convs,self.off_diag_convs,self.cross_ch_convs)
    def forward(self, decodings):
        """
        Perform decoding, and here approximate supn parameters from second layer(after bottleneck).
        """
        final_res = decodings[-1].shape[-2:]  # Resolution of the final decoding layer
        results = []



        # Start from the second element (decodings[1]) onwards
        for idx, (ip_encoding, pre_conv, log_diag_conv, off_diag_conv) in enumerate(
                zip(decodings[1::], self.pre_convs[::-1], self.log_diag_convs[::-1], self.off_diag_convs[::-1])):

            # Apply pre-conv layer
            ip_encoding = self.do_layer(ip_encoding)
            ip_encoding = torch.nn.functional.leaky_relu(ip_encoding)
            ip_encoding = pre_conv(ip_encoding)
            ip_encoding = torch.nn.functional.leaky_relu(ip_encoding)

            # Log diagonal prediction
            if self.predict_log_diags:
                log_diag = log_diag_conv(ip_encoding)
            else:
                log_diag = torch.log(torch.square(log_diag_conv(ip_encoding))) * 0.5

            # Off diagonal prediction
            off_diag = off_diag_conv(ip_encoding)
            

            # Cross-channel prediction
            cross_ch = None
            if self.cross_ch_convs:
                cross_ch = self.cross_ch_convs[idx](ip_encoding)

            # Smoothing for intermediate layers
            log_diag, off_diag = self.apply_smoothing(log_diag, off_diag)

            results.append((log_diag, off_diag, cross_ch))

        return results


class MeanDecoder(nn.Module):
    def __init__(self, num_init_features, max_ch, num_log_diags, use_bias=True, num_scales = 4, dropout_rate=0.0):
        """
        Construct a decoder for mean only - using skip connections from encoder, and from last layer
        """
        super().__init__()
    

        conv_class = nn.Conv2d

        # Per-layer convolutional prediction blocks
        self.pre_convs = nn.ModuleList()

        self.mean_convs = nn.ModuleList()

        self.do_layer = torch.nn.Dropout(p=dropout_rate)


        for level in range(num_scales):  # Loop backward (from largest features)
            features = min(max_ch, num_init_features * pow(2, level))

            # Pre-conv layer for processing input
            self.pre_convs.append(conv_class(features, features, kernel_size=3, padding=1,
                                             padding_mode='reflect', bias=use_bias))

            # Prediction layers
            self.mean_convs.append(conv_class(features, num_log_diags, kernel_size=3, padding=1,
                                              padding_mode='reflect', bias=use_bias))


    def forward(self, decodings):
        """
        Perform decoding, and here approximate mean from second layer(after bottleneck).
        """
        final_res = decodings[-1].shape[-2:]  # Resolution of the final decoding layer
        results = []


        # Start from the second element (decodings[1]) onwards
        for idx, (ip_encoding, pre_conv, mean_conv) in enumerate(
                zip(decodings[1::], self.pre_convs[::-1], self.mean_convs[::-1])):

            # Apply pre-conv layer
            ip_encoding = self.do_layer(ip_encoding)
            ip_encoding = torch.nn.functional.leaky_relu(ip_encoding)
            ip_encoding = pre_conv(ip_encoding)
            ip_encoding = torch.nn.functional.leaky_relu(ip_encoding)


            # Mean prediction
            mean = mean_conv(ip_encoding)

            results.append((mean))

        return results



class SUPNEncoderDecoder(nn.Module):
    def __init__(self, in_channels: int, num_out_ch: int, num_init_features: int, num_log_diags: int,
                 num_local_connections: list, use_skip_connections: bool, num_scales: int, num_prediction_scales: int,
                 max_dropout=0.0, use_bias=False, use_3D=False, max_ch=1024):
        """
        Construct an encoder decoder network for a SUPN prediction model
        Args:
            in_channels: The number of image input arguments (e.g. 3 for an RGB image)
            num_init_features: The initial number of feature channels
            num_out_ch: The number of output channels (e.g. 1 for binary segmentation)
            num_log_diags: The number of output logvariances (e.g. 1 for a single channel image)
            num_local_connections: The local connection distance for the SUPN matrix
            use_skip_connections: Whether to use skip connections in the unet or not
            num_scales: the number of levels/scalings in the unet
        """

        assert len(num_local_connections) == num_prediction_scales
        
        super().__init__()
        self.use_3d = use_3D
        self.local_connection_dist = num_local_connections
        num_off_diags = [get_num_off_diag_weights(i, use_3D) * in_channels for i in self.local_connection_dist]
        self.num_scales = num_scales + 1  # Add one for the bottleneck layer
        self.num_prediction_scales = num_prediction_scales





        # Encoder-decoder backbone
        self.encoder_decoder = ResNetUnet2Decoders(in_channels, num_init_features=num_init_features,
                                          use_skip_connections=use_skip_connections, num_scales=self.num_scales,
                                          max_dropout=max_dropout, use_bias=use_bias, use_3d=use_3D,
                                          max_ch=max_ch)

        # SUPN decoder with dynamic per-level prediction blocks
        self.supn_decoder = SUPNDecoderDirect(min(num_init_features, max_ch),
                                              num_log_diags, num_off_diags, 
                                              num_cross_ch_images=None if num_log_diags <= 1 else (num_log_diags ** 2 - num_log_diags) // 2, # (self.log_diag.shape[1]**2 - self.log_diag.shape[1]) // 2
                                              use_bias=use_bias, use_3D=use_3D, num_scales=self.num_prediction_scales, max_ch=max_ch)
        
        self.mean_decoder = MeanDecoder(min(num_init_features, max_ch),num_log_diags = num_log_diags, max_ch=max_ch,
                                        num_scales=self.num_prediction_scales, use_bias=use_bias)
        
        

    def forward(self, x):
        """
        Forward pass: Encodes input and decodes predictions for all levels.
        """
        from supn_base.supn_data import SUPNData

        # Encode and decode features
        final_encoding, enc_dec = self.encoder_decoder(x)
        diff = (self.num_scales - self.num_prediction_scales) - 1

        results = []
        # Decode predictions for each level
        mean_decodings = self.mean_decoder(enc_dec['decodings_mean'][diff::])
        supn_decodings = self.supn_decoder(enc_dec['decodings_supn'][diff::])

        mean_decodings = mean_decodings[::-1]
        supn_decodings = supn_decodings[::-1]
        i = 0
        for mean, supn in zip(mean_decodings,supn_decodings):
            log_diag, off_diag, cross_ch = supn
            ## currently I dont have access to the supn cholespy so solver does not work
            results.append(torch.distributions.SUPN(SUPNData(mean = mean, log_diag=log_diag, off_diag=off_diag, cross_ch=cross_ch,
                                                 local_connection_dist=self.local_connection_dist[i])))
            # results.append(SUPNData(mean = mean, log_diag=log_diag, off_diag=off_diag, cross_ch=cross_ch,
            #                                      local_connection_dist=self.local_connection_dist[i]))
            i += 1
        
        return results