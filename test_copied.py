# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Blocks of networks
import torch.nn as nn
from enum import Enum


class ResNetType(Enum):
    UPSAMPLE = 0
    DOWNSAMPLE = 1
    SAME = 2


class ResnetBlock(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, block_type, use_bnorm=True, use_bias=True, layers_per_group=1, use_3D=False):
        super().__init__()
        assert isinstance(block_type, ResNetType)
        self.type = block_type

        self._use_bnorm = use_bnorm

        if use_3D:
            self._conv1 = nn.Conv3d(int(in_channels), int(out_channels), 3, bias=use_bias, padding=1)
            self._conv2 = nn.Conv3d(int(out_channels), int(out_channels), 3, bias=use_bias, padding=1)
        else:
            self._conv1 = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=use_bias, padding=1)
            self._conv2 = nn.Conv2d(int(out_channels), int(out_channels), 3, bias=use_bias, padding=1)

        if self._use_bnorm:
            self._bn1 = nn.GroupNorm(in_channels//layers_per_group, in_channels)#nn.BatchNorm2d(in_channels)
            self._bn2 = nn.GroupNorm(out_channels//layers_per_group, out_channels)#nn.BatchNorm2d(out_channels)

        self._relu1 = nn.LeakyReLU(0.2)
        self._relu2 = nn.LeakyReLU(0.2)
        self._resample = None

        if block_type == ResNetType.UPSAMPLE:
            if use_3D:
                self._resample = nn.Upsample(scale_factor=2, mode='trilinear')
            else:
                self._resample = nn.UpsamplingBilinear2d(scale_factor=2)

        elif block_type == ResNetType.DOWNSAMPLE:
            if use_3D:
                self._resample = nn.AvgPool3d(2)
                self._resample2 = nn.AvgPool3d(2)
            else:
                self._resample = nn.AvgPool2d(2)
                self._resample2 = nn.AvgPool2d(2)

        if use_3D:
            self._resid_conv = nn.Conv3d(int(in_channels), int(out_channels), 1, bias=use_bias)
        else:
            self._resid_conv = nn.Conv2d(int(in_channels), int(out_channels), 1, bias=use_bias)

    def forward(self, input):
        resid_connection = None

        if self.type == ResNetType.UPSAMPLE or self.type == ResNetType.SAME:
            if self._resample:
                input = self._resample(input)
            resid_connection = self._resid_conv(input)

        net = input
        if self._use_bnorm:
            net = self._bn1(net)
        net = self._relu1(net)

        # print(f"Shape before convolution: {net.shape}")

        net = self._conv1(net)
        # print(f"Shape after convolution: {net.shape}")


        if self._use_bnorm:
            net = self._bn2(net)
        net = self._relu2(net)
        net = self._conv2(net)

        if self.type == ResNetType.DOWNSAMPLE:
            net = self._resample(net)
            resid_connection = self._resid_conv(input)
            resid_connection = self._resample2(resid_connection)

        net = net + resid_connection
        return net
    
from dataclasses import dataclass
import torch

#initial data class to hold all of the supn dist parameters.

@dataclass
class Parent:
    def __post_init__(self):
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")


@dataclass
class SUPNData(Parent):
    mean: torch.Tensor
    log_diag: torch.Tensor
    off_diag: torch.Tensor
    cross_ch: torch.Tensor = None
    local_connection_dist: int = 2
    use_3d: bool = False

    def __post_init__(self):
        if self.use_3d:
            assert self.mean.ndim == 5
            assert self.log_diag.ndim == 5
            assert self.off_diag.ndim == 5
        else:
            assert self.mean.ndim == 4
            assert self.log_diag.ndim == 4
            assert self.off_diag.ndim == 4

        assert self.mean.shape == self.log_diag.shape
        assert self.mean.device == self.log_diag.device == self.off_diag.device

    def get_num_ch(self):
        return self.log_diag.shape[1]

    def test_consistency(self):
        """
        Test for consistency of the data
        """
        if self.use_3d:
            assert self.log_diag.ndim == 5
            assert self.off_diag.ndim == 5
            num_F = self.off_diag.shape[-4]
        else:
            assert self.log_diag.ndim == 4
            assert self.off_diag.ndim == 4
            num_F = self.off_diag.shape[-3]

        assert self.log_diag.device == self.off_diag.device

        assert self.log_diag.shape[1] > 0

        if self.cross_ch is not None:
            assert self.cross_ch.shape[1] == (self.log_diag.shape[1]**2 - self.log_diag.shape[1]) // 2
            assert self.off_diag.shape[1] == self.log_diag.shape[1] * get_num_off_diag_weights(self.local_connection_dist, self.use_3d)


def get_num_off_diag_weights(local_connection_dist, use_3d=False):
    """Returns the number of off-diagonal entries required for a particular sparsity.

    Args:
        local_connection_dist: Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_3d: Create 3D filters (i.e. 3x3x3) rather than 2D. Defaults to False.

    Returns:
        num_weigts_required (int)
    """
    filter_size = 2 * local_connection_dist + 1
    if use_3d:
        filter_size_dims = filter_size * filter_size * filter_size
    else:
        filter_size_dims = filter_size * filter_size
    filter_size_dims_2 = filter_size_dims // 2
    return filter_size_dims_2


def get_num_cross_channel_weights(num_channels):
    """Returns the number of cross-channel weights required for a particular number of channels.
    This connects each channel to every other one.

    Args:
        num_channels: The number of channels.

    Returns:
        num_weights_required (int)
    """
    return (num_channels**2 - num_channels) // 2


def convert_log_to_diag_weights(log_diag_weights):
    """Converts the log weight values into the actual positive diagonal values.

    Args:
        log_diag_weights(tensor): [BATCH x 1 x W x H] log of the diagonal terms (mapped through exp).

    Returns:
        diag_weights(tensor): [BATCH x 1 x W x H] actual weights (guaranteed positive)
    """
    diag_values = torch.exp(log_diag_weights)
    return diag_values



# NDFC Jan 2021 - Toolkit functions to deal with sparse precision in Cholesky form.
#
# NOTES:
#   - At the moment probably want to use the use_transpose=True setting for all the functions - mixing modes
#     is not advised as these are not strict transposes of one-another..
#

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F
import scipy.sparse as sparse
from typing import Tuple

# from supn_data import convert_log_to_diag_weights, get_num_off_diag_weights, get_num_cross_channel_weights


def build_off_diag_filters(local_connection_dist, use_transpose=True, device=None, dtype=torch.float, use_3d=False):
    """Create the conv2d/conv3d filter weights for the off-diagonal components of the sparse chol.

    NOTE: Important to specify device if things might run under cuda since constants are created and need to be
        on the correct device.

    Args:
        local_connection_dist (int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose (bool): Defaults to True - usually what we want for the jacobi sampling.
        device: Specify the device to create the constants on (i.e. cpu vs gpu).
        dtype: Specify the dtype to use - defaults to torch.float.
        use_3d (bool): Create 3D filters (i.e. 3x3x3) rather than 2D. Defaults to False.

    Returns:
        tri_off_diag_filters (tensor): [num_off_diag_weights x 1 x [F if use_3d] x F x F] Conv2d/3d kernel filters.
            (Where F = filter_size)
    """
    filter_size = 2 * local_connection_dist + 1
    filter_size_dims_2 = get_num_off_diag_weights(local_connection_dist, use_3d=use_3d)

    if use_transpose:
        tri_off_diag_filters = torch.cat((torch.zeros(filter_size_dims_2, (filter_size_dims_2 + 1),
                                                      device=device, dtype=dtype),
                                          torch.eye(filter_size_dims_2,
                                                    device=device, dtype=dtype)), dim=1)
    else:
        tri_off_diag_filters = torch.cat((torch.fliplr(torch.eye(filter_size_dims_2,
                                                                 device=device, dtype=dtype)),
                                          torch.zeros(filter_size_dims_2, (filter_size_dims_2 + 1),
                                                      device=device, dtype=dtype)), dim=1)

    if use_3d:
        tri_off_diag_filters = torch.reshape(tri_off_diag_filters, (filter_size_dims_2, 1, filter_size, filter_size, filter_size))
    else:
        tri_off_diag_filters = torch.reshape(tri_off_diag_filters, (filter_size_dims_2, 1, filter_size, filter_size))

    return tri_off_diag_filters


def apply_off_diag_weights_offset(off_diag_weights,
                                  local_connection_dist,
                                  use_transpose=True,
                                  reverse_direction=False,
                                  use_3d=False):
    """Shuffle the off-diagonal weights based on the filters to perform the transpose operation.

    Parameters:
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.
        reverse_direction(bool): Use the reverse direction for undoing the operation (default False).
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        shuffled_off_diag_weights(tensor): [B ? 1 x F x [D] x W x H].
    """
    if use_3d:
        assert off_diag_weights.ndim == 5
    else:
        assert off_diag_weights.ndim == 4

    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=not use_transpose,
                                                  use_3d=use_3d)

    channel_filt = torch.split(tri_off_diag_filters, 1, dim=0)

    assert all([torch.nonzero(c).shape[0] == 1 for c in channel_filt])

    if use_3d:
        fD, fW, fH = tri_off_diag_filters.shape[2:]

        im_size_h = off_diag_weights.shape[-1]
        im_size_w = off_diag_weights.shape[-2]

        fD_mid = fD // 2
        fW_mid = fW // 2
        fH_mid = fH // 2

        offset_dim = -4

        def map_idx(idx):
            shuff_idx = - (((idx[0] - fD_mid) * im_size_w * im_size_h) + ((idx[1] - fW_mid) * im_size_h) + (
                        idx[2] - fH_mid))
            if reverse_direction:
                shuff_idx = - shuff_idx
            return shuff_idx

        indices = [map_idx(torch.nonzero(c)[0, 2:]).item() for c in channel_filt]
    else:
        fW, fH = tri_off_diag_filters.shape[2:]

        im_size_h = off_diag_weights.shape[-1]

        fW_mid = fW // 2
        fH_mid = fH // 2

        offset_dim = -3

        def map_idx(idx):
            shuff_idx = - (((idx[0] - fW_mid) * im_size_h) + (idx[1] - fH_mid))
            if reverse_direction:
                shuff_idx = - shuff_idx
            return shuff_idx

        indices = [map_idx(torch.nonzero(c)[0, 2:]).item() for c in channel_filt]

    # if reverse_direction:
    #     assert all([i < 0 for i in indices])
    # else:
    #     assert all([i > 0 for i in indices])

    channel_weights = torch.split(off_diag_weights, 1, dim=offset_dim)
    channel_weights = [torch.roll(cw, offset) for cw, offset in zip(channel_weights, indices)]

    off_diag_weights_shuffled = torch.cat(channel_weights, dim=offset_dim)

    return off_diag_weights_shuffled


def get_prec_chol_as_sparse_tensor(log_diag_weights,
                                   off_diag_weights,
                                   local_connection_dist,
                                   use_transpose=True,
                                   use_3d=False,
                                   cross_ch=None):
    """Returns the precision Cholesky matrix as a sparse COO tensor (on the CPU).

    Args:
        log_diag_weights(tensor): [BATCH x CH x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [BATCH x CH x F x W x H] off-diagonal terms.
                                  F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist (int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose (bool): Defaults to True.
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        sparse_prec_chol (torch.sparse_coo_tensor): The Cholesky factor as a sparse COO precision matrix.
    """
    if use_3d:
        assert log_diag_weights.ndim == 5
        assert off_diag_weights.ndim == 5
    else:
        assert log_diag_weights.ndim == 4
        assert off_diag_weights.ndim == 4

    assert log_diag_weights.device == off_diag_weights.device
    device = off_diag_weights.device
    dtype = off_diag_weights.dtype

    num_ch = log_diag_weights.shape[1]

    # # The sparse tensor package requires the use of doubles..
    # dtype = torch.double

    num_off_diag_weights = get_num_off_diag_weights(local_connection_dist=local_connection_dist,
                                                    use_3d=use_3d)
    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  device=device,
                                                  dtype=dtype,
                                                  use_3d=use_3d)

    #TODO: Check this is OK..
    if not use_transpose:
        off_diag_weights = apply_off_diag_weights_offset(off_diag_weights=off_diag_weights,
                                                         local_connection_dist=local_connection_dist,
                                                         use_transpose=not use_transpose,
                                                         use_3d=use_3d)

    batch_size = log_diag_weights.shape[0]


    if use_3d:
        im_size_D = log_diag_weights.shape[2]
        im_size_H = log_diag_weights.shape[3]
        im_size_W = log_diag_weights.shape[4]

        im_all_size = im_size_D * im_size_H * im_size_W
        view_dims = (1, 1, im_size_D, im_size_H, im_size_W)
        cat_dim = -4
    else:
        im_size_H = log_diag_weights.shape[2]
        im_size_W = log_diag_weights.shape[3]

        im_all_size = im_size_H * im_size_W
        view_dims = (1, 1, im_size_H, im_size_W)
        cat_dim = -3

    diag_values = convert_log_to_diag_weights(log_diag_weights)

    index_input = torch.arange(im_all_size, dtype=dtype, device=device).view(*view_dims) + 1

    # The following is involved and probably not the most efficient way of doing things,
    # it was more focused on being correct (which I think it is!). Essentially we need
    # to determine the batch, row and column indices of the sparse values..

    indices_col = []
    indices_row = []
    values = []


    if use_3d:
        off_diag_indices = F.conv3d(index_input.view(-1, 1, im_size_D, im_size_H, im_size_W).double(),
                                    tri_off_diag_filters.double(),
                                    padding=local_connection_dist, stride=1)
    else:
        off_diag_indices = F.conv2d(index_input.view(-1, 1, im_size_H, im_size_W).double(),
                                    tri_off_diag_filters.double(),
                                    padding=local_connection_dist, stride=1)
    # Repeat the process for the multiple channels that we have in the data, adding the channel offset
    # We need to mask the off-diagonal indices to ensure that we don't add extra non-zero values
    off_diag_indices_mask = off_diag_indices > 0
    for ch in range(num_ch):
        ch_offset = ch*(im_all_size)
        indices_col.extend([index_input+ch_offset, (off_diag_indices+ch_offset)*off_diag_indices_mask])
        indices_row.extend((1 + num_off_diag_weights) * [(index_input+ch_offset)])
        values.extend([diag_values[:, ch:ch+1, :, :], off_diag_weights[:, ch*num_off_diag_weights:num_off_diag_weights*(ch+1), :, :]])


    # Now need to add the cross-channel weights
    cross_ch_idx = 0
    for ch in range(num_ch-1):
        for ch2 in range(ch+1, num_ch):
            indices_row.append(index_input+ch*im_all_size)
            indices_col.append(index_input+ch2*im_all_size)
            values.append(cross_ch[:, cross_ch_idx:cross_ch_idx+1,...])
            cross_ch_idx += 1


    all_indices_col = torch.cat(indices_col, dim=cat_dim)
    all_indices_row = torch.cat(indices_row, dim=cat_dim)
    all_values = torch.cat(values, dim=cat_dim)

    all_indices_col = all_indices_col.flatten().long()
    all_indices_row = all_indices_row.flatten().long()
    all_values = all_values.flatten()

    all_indices_col_used = all_indices_col[all_indices_col > 0]
    all_indices_row_used = all_indices_row[all_indices_col > 0]
    all_values_used = all_values[all_indices_col.repeat(batch_size).flatten() > 0]

    all_indices_col_used -= 1
    all_indices_row_used -= 1

    all_indices_batch_used = torch.arange(batch_size, device=device).view(-1, 1).expand(batch_size,
                                                                                        all_indices_row_used.shape[0])

    all_indices_batch_used = all_indices_batch_used.flatten()

    all_indices_row_used = all_indices_row_used.repeat(batch_size).flatten()
    all_indices_col_used = all_indices_col_used.repeat(batch_size).flatten()

    sparse_LT_coo = torch.sparse_coo_tensor(indices=torch.stack([all_indices_batch_used, all_indices_row_used, all_indices_col_used]),
                                            values=all_values_used,
                                            size=[batch_size, im_all_size*num_ch, im_all_size*num_ch],
                                            dtype=torch.double)

    return sparse_LT_coo


def apply_sparse_chol_rhs_matmul(dense_input,
                                 log_diag_weights,
                                 off_diag_weights,
                                 local_connection_dist,
                                 use_transpose=True,
                                 use_3d=False,
                                 cross_ch=None):
    """Apply the sparse chol matrix to a dense input on the rhs i.e. result^T = input^T L  (standard matrix mulitply).

    IMPORTANT: Only valid for a single channel at the moment.

    Args:
        dense_input(tensor): [BATCH x 1 x W x H] Input matrix (must be single channel).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        product(tensor): [BATCH x 1 x W x H] Result of (L dense_input) or (L^T dense_input).
    """
    if use_3d:
        assert dense_input.ndim == 5
        assert log_diag_weights.ndim == 5
        assert off_diag_weights.ndim == 5
    else:
        assert dense_input.ndim == 4
        assert log_diag_weights.ndim == 4
        assert off_diag_weights.ndim == 4

    # Check how many channels
    num_ch = dense_input.shape[1]

    assert log_diag_weights.dtype == off_diag_weights.dtype
    assert dense_input.dtype == log_diag_weights.dtype

    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  device=dense_input.device,
                                                  dtype=log_diag_weights.dtype,
                                                  use_3d=use_3d)

    # need to copy the weights based on the number of channels
    tri_off_diag_filters = torch.cat([tri_off_diag_filters for i in range(num_ch)], dim=0)

    # TODO: Need to resolve the issue around whether or not to use apply_off_diag_weights_offset here for lower mode...
    # assert use_transpose is True

    diag_values = convert_log_to_diag_weights(log_diag_weights)

    if use_3d:
        interim = F.conv3d(dense_input, tri_off_diag_filters, padding=local_connection_dist, stride=1, groups=num_ch)
        interim = interim.view(-1, num_ch, interim.shape[1]//num_ch, interim.shape[-3], interim.shape[-2], interim.shape[-1])
        after_weights = torch.einsum('bqfdwh, bqfdwh->bqdwh' if off_diag_weights.shape[0] > 1 else 'bqfdwh, xqfdwh->bqdwh',
                                     interim, off_diag_weights.view((-1,) + interim.shape[1:]))
    else:
        # Use a grouped convolution to apply the replicated off-diag filters separately across channels
        interim = F.conv2d(dense_input, tri_off_diag_filters, padding=local_connection_dist, stride=1, groups=num_ch)
        interim = interim.view(-1, num_ch, interim.shape[1]//num_ch, interim.shape[2], interim.shape[3])

        # Do a channelwise multiply and sum with the off-diagonal weights
        after_weights = torch.einsum('bqfwh, bqfwh->bqwh' if off_diag_weights.shape[0] > 1 else 'bqfwh, xqfwh->bqwh',
                                    interim, off_diag_weights.view((-1,) + interim.shape[1:]))



    result = diag_values * dense_input + after_weights.view(*dense_input.shape)

    # If we have multiple channels and cross channel weights, we need to incorporate these
    if num_ch > 1 and cross_ch is not None:
        # Make our cross channel filters
        cross_channel_filters = torch.zeros((get_num_cross_channel_weights(num_ch), num_ch, 1, 1), device=dense_input.device, dtype=dense_input.dtype, requires_grad=False)
        if use_3d:
            cross_channel_filters = cross_channel_filters.unsqueeze(-1)
        filter_split_idx = []
        with torch.no_grad():
            if use_transpose:
                # These should be of shape [num_cross_ch x C x 1 x 1], containing a 1 where channel j is connected to channel i (i < j)
                filter_idx = 0
                for i in range(num_ch):
                    for j in range(i+1, num_ch):
                        cross_channel_filters[filter_idx, j, ...] = 1
                        filter_idx = filter_idx + 1
                    filter_split_idx.append(filter_idx)
            else:
                filter_idx = 0
                for i in range(num_ch):
                    for j in range(0, i):
                        cross_channel_filters[filter_idx, j, ...] = 1
                        filter_idx = filter_idx + 1
                    filter_split_idx.append(filter_idx)




        # Pick out the right channel information - this is a 1x1 convolution with a single 1 for the correct channel
        if use_3d:
            interim = F.conv3d(dense_input, cross_channel_filters, stride=1)
        else:
            interim = F.conv2d(dense_input, cross_channel_filters, stride=1)


        # Multiply elementwise by the cross channel weights
        weighted_interim = interim * cross_ch

        # We will have different numbers of entries per channel, so we need to split the result and sum them individually.
        channelwise_interim = torch.tensor_split(weighted_interim, filter_split_idx, dim=1)

        # Sum over the number of filters per channel for each channel, the last should be 0
        channelwise_effect = torch.cat([torch.sum(channelwise_interim[i], dim=1, keepdim=True) for i in range(num_ch)], dim=1)

        # Add the result to the previous result
        result = result + channelwise_effect



    return result


def log_prob_from_sparse_chol_prec_with_whitened_mean(x,
                                                      whitened_mean,
                                                      log_diag_weights,
                                                      off_diag_weights,
                                                      local_connection_dist,
                                                      use_transpose=True):
    """Calculate the log probability of x under the "whitened mean" and precision Cholesky.

        The whitened mean is defined as (L^T mu) so the real mean needs to be found by
        solving for (L^T)^-1 (L^T mu) = mu.

    Args:
        x(tensor): [BATCH x 1 x W x H] Data, i.e. return log p(x).
        whitened_mean(tensor): [BATCH x 1 x W x H] the whitened mean (L^T mu).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.

    Returns:
        log_prob(tensor): [BATCH] The log prob of each element in the batch.
    """
    assert log_diag_weights.ndim == 4
    assert off_diag_weights.ndim == 4

    assert log_diag_weights.shape[1] == 1
    im_size_w = log_diag_weights.shape[2]
    im_size_h = log_diag_weights.shape[3]

    fitting_term = apply_sparse_chol_rhs_matmul(x,
                                                log_diag_weights=log_diag_weights,
                                                off_diag_weights=off_diag_weights,
                                                local_connection_dist=local_connection_dist,
                                                use_transpose=use_transpose)

    fitting_term = fitting_term - whitened_mean

    constant_term = im_size_w * im_size_h * torch.log(torch.Tensor([2.0]) * np.pi)
    constant_term = constant_term.to(log_diag_weights.device)

    # Can't do this in case we do something funning to the diag weights..
    log_det_term = 2.0 * torch.sum(log_diag_weights, dim=(1,2,3,)) # Note these are precision NOT covariance L

    # Do this in case we do something funny in the conversion..
    actual_log_diag_values = torch.log(convert_log_to_diag_weights(log_diag_weights))
    log_det_term = 2.0 * torch.sum(actual_log_diag_values, dim=(1, 2, 3,))  # Note these are precision NOT covariance L

    log_prob = -0.5 * torch.sum(torch.square(fitting_term), dim=(1,2,3,)) \
               -0.5 * constant_term \
               +0.5 * log_det_term # Note positive since precision..

    return log_prob


def log_prob_from_sparse_chol_prec(x,
                                   mean,
                                   log_diag_weights,
                                   off_diag_weights,
                                   local_connection_dist,
                                   use_transpose=True,
                                   use_3d=False,
                                   cross_ch=None):
    """Calculate the log probability of x under the mean and precision Cholesky.

    Args:
        x(tensor): [BATCH x 1 x W x H] Data, i.e. return log p(x).
        mean(tensor): [BATCH x 1 x W x H] the mean (mu).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        log_prob(tensor): [BATCH] The log prob of each element in the batch.
    """
    if use_3d:
        assert x.ndim == 5
        assert mean.ndim == 5
        assert log_diag_weights.ndim == 5
        assert off_diag_weights.ndim == 5

        assert x.shape[-4:] == mean.shape[-4:]
        assert log_diag_weights.shape[2:] == x.shape[2:]

        im_size_d = log_diag_weights.shape[2]
        im_size_w = log_diag_weights.shape[3]
        im_size_h = log_diag_weights.shape[4]

        all_size = im_size_d * im_size_w * im_size_h
        dims_to_sum = (1, 2, 3, 4,)
    else:
        assert log_diag_weights.ndim == 4
        assert off_diag_weights.ndim == 4

        im_size_w = log_diag_weights.shape[2]
        im_size_h = log_diag_weights.shape[3]

        all_size = im_size_w * im_size_h
        dims_to_sum = (1, 2, 3,)

    assert use_transpose
    fitting_term = apply_sparse_chol_rhs_matmul(x - mean,
                                                log_diag_weights=log_diag_weights,
                                                off_diag_weights=off_diag_weights,
                                                local_connection_dist=local_connection_dist,
                                                use_transpose=use_transpose,
                                                use_3d=use_3d,
                                                cross_ch=cross_ch)

    constant_term = all_size * torch.log(torch.Tensor([2.0]) * np.pi)
    constant_term = constant_term.to(log_diag_weights.device)

    # Can't do this in case we do something funny to the diag weights...
    # log_det_term = 2.0 * torch.sum(log_diag_weights, dim=dims_to_sum) # Note these are precision NOT covariance L

    # Do this in case we do something funny in the conversion...
    actual_log_diag_values = torch.log(convert_log_to_diag_weights(log_diag_weights))
    log_det_term = 2.0 * torch.sum(actual_log_diag_values, dim=dims_to_sum)  # Note these are precision NOT covariance L

    log_prob = -0.5 * torch.sum(torch.square(fitting_term), dim=dims_to_sum) \
               -0.5 * constant_term \
               +0.5 * log_det_term # Note positive since precision..

    return log_prob

def coo_to_supn_sparse(coo_prec_chol: torch.sparse_coo_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Method to convert a sparse tensor in coo format to a supn formatted
    precision cholesky. Tested on square images for single distribution supn models
    Assumes the input is a square matrix with a single channel

    Parameters
    ----------
    coo_prec_chol : torch.sparse.COO
    sparse tensor in coo format
    Returns
    -------
    log_diag_weights : torch.Tensor
    log diagonal weights in supn format
    off_diag_weights : torch.Tensor
    off diagonal weights in supn format
    '''
    im_w = int(torch.sqrt(torch.tensor(coo_prec_chol.shape[1])))
    im_h = im_w

    diagonals = []
    for i in range(coo_prec_chol.shape[1]):
        if (coo_prec_chol[0].to_dense().diagonal(offset=i)**2).sum() != 0:
            diagonals.append(torch.cat([coo_prec_chol[0].to_dense().diagonal(offset=i),torch.zeros(i,device='cuda')]))
    supn_chol = torch.stack(diagonals)
    log_diag_weights = torch.log(supn_chol[0])
    off_diag_weights = supn_chol[1:]
    return log_diag_weights.reshape([1,1,im_w,im_h]), off_diag_weights.reshape([1,-1,im_w,im_h])


import sys
sys.path.append('../')

import torch
from torch.distributions import Distribution
# from sparse_precision_cholesky import get_prec_chol_as_sparse_tensor, \
#     log_prob_from_sparse_chol_prec

# from cholespy_solver import SUPNSolver, sparse_chol_linsolve, sample_zero_mean
# from supn_data import SUPNData, get_num_off_diag_weights

class SUPN(Distribution):
    """
    SUPN distribution class implementing the torch.distributions.Distribution interface.
    """

    def __init__(self,
                 supn_data: SUPNData):
        self.supn_data = supn_data
        # self.supn_solver = SUPNSolver(supn_data)
        super(SUPN, self).__init__()


    # def sample(self,
    #            num_samples: int = 1) -> torch.Tensor:

    #     zero_mean_sample = sample_zero_mean(self.supn_data, num_samples, self.supn_solver)

    #     return  zero_mean_sample + self.mean

    def log_prob(self, data: torch.tensor, stop_grads: bool = False) -> torch.Tensor:
        """
        Compute the log probability of the data given the distribution.
        stop_grads: blocks gradients back into the supn parameters
        """
        if stop_grads:
            return log_prob_from_sparse_chol_prec(x = data,
                                       mean = self.supn_data.mean.detach(),
                                       log_diag_weights = self.supn_data.log_diag.detach(),
                                       off_diag_weights = self.supn_data.off_diag.detach(),
                                       local_connection_dist = self.supn_data.local_connection_dist,
                                       use_transpose=True,
                                       use_3d=self.supn_data.use_3d,
                                       cross_ch=self.supn_data.cross_ch.detach())
        else:
            return log_prob_from_sparse_chol_prec(x = data,
                                       mean = self.supn_data.mean,
                                       log_diag_weights = self.supn_data.log_diag,
                                       off_diag_weights = self.supn_data.off_diag,
                                       local_connection_dist = self.supn_data.local_connection_dist,
                                       use_transpose=True,
                                       use_3d=self.supn_data.use_3d,
                                       cross_ch=self.supn_data.cross_ch)


    def cov(self) -> torch.Tensor:
        # some methods to get the covariance matrix from chol or prec (or maybe just a row)
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        return self.supn_data.mean

    @property
    def precision(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def precision_cholesky(self) -> torch.Tensor:
        # Return the precision is a sparse COO torch tensor
        return get_prec_chol_as_sparse_tensor(log_diag_weights = self.supn_data.log_diag,
                                    off_diag_weights = self.supn_data.off_diag,
                                    local_connection_dist = self.supn_data.local_connection_dist,
                                    use_transpose = True,
                                    use_3d = self.supn_data.use_3d,
                                    cross_ch=self.supn_data.cross_ch)


# Register the distribution
torch.distributions.SUPN = SUPN

# Adapted code from Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk),
# with the following key changes and adjustments:
# -> The model now infers SUPN (Sparse Uncertainty Prediction Network) objects at multiple intermediate layers,
#    not just the last layer, with adaptive control via the `nr_of_prediction_scales`.
# -> Outputs the SUPN objects as a list of layers, each with corresponding scalings.
# -> For each SUPN object, the `nr_of_connections` (i.e., local connection distances) is provided,
#    and it must match the length of the scaling levels (`nr_of_scaling`).

## currently adapted from Paula's code

# import sys
# sys.path.append('../')


import torch
import torch.nn as nn
from enum import Enum

# from supn_base.sparse_precision_cholesky import get_num_off_diag_weights
# from supn_base.cholespy_solver import get_num_off_diag_weights
# from supn_base.supn_data import SUPNData


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

            # print(f"ip encodings {ip_encoding.shape}")

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

            # print(f"Mean: {mean.shape}")

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
        # from supn_base.supn_data import SUPNData

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
    

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BRATSDataset(Dataset):
    def __init__(self, flair_dir='dataset/2afc', masks_dir='dataset/2afc', dataset_type="train", img_size=(256, 256), num_masks=5, split_ratio=0.8):
        """
        FLAIR dataset with corresponding masks from an ensemble.
        The dataset will load both FLAIR images and 5 masks corresponding to each FLAIR image.

        :param flair_dir: Root directory containing FLAIR images.
        :param masks_dir: Root directory containing mask images.
        :param dataset_type: 'train' or 'val' to specify which split to use.
        :param img_size: Target size of the images.
        :param num_masks: Number of masks to load per FLAIR image.
        :param split_ratio: Ratio to split dataset into train and validation (e.g., 0.8 means 80% train, 20% validation).
        """
        super(BRATSDataset, self).__init__()
        self.flair_dir = flair_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.num_masks = num_masks
        self.split_ratio = split_ratio

        # Get list of all FLAIR images
        self.flair_filenames = self._get_image_filenames(self.flair_dir)

        # Split into train and validation based on split_ratio
        num_train = int(len(self.flair_filenames) * self.split_ratio)
        random.shuffle(self.flair_filenames)

        # Assign images to train/val based on the split ratio
        if dataset_type == "train":
            self.flair_filenames = self.flair_filenames[:num_train]
        elif dataset_type == "val":
            self.flair_filenames = self.flair_filenames[num_train:]
        else:
            raise ValueError("dataset_type should be either 'train' or 'val'.")

    def _get_image_filenames(self, directory):
        """
        Get all image filenames (excluding directories) from the given folder.
        """
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def _get_mask_filenames(self, flair_filename):
        """
        For each FLAIR image, get corresponding mask filenames.
        Assumes the masks are named based on the FLAIR image with a mask index (e.g., flair_image_1_mask_1.png).
        """
        base_name = os.path.splitext(os.path.basename(flair_filename))[0]
        mask_filenames = []
        # for i in range(1, self.num_masks + 1):
        for i in range(1, 5):
            mask_filename = os.path.join(self.masks_dir, f"{base_name}_ensemble_{i}.png")
            if not mask_filename.endswith("_ensemble_2.png"):  # Exclude _ensemble_2
                mask_filenames.append(mask_filename)
        return mask_filenames

        # FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/Diss/ensemble_predictions/patient_80_9_mask_1.png'


    def __len__(self):
        return len(self.flair_filenames)

    def __getitem__(self, index):
        """
        Load FLAIR image and corresponding masks.
        """
        # Load FLAIR image
        flair_image_path = self.flair_filenames[index]
        flair_image = Image.open(flair_image_path).convert("L")

        # Load masks corresponding to this FLAIR image
        mask_filenames = self._get_mask_filenames(flair_image_path)
        masks = [Image.open(mask_filename).convert("L") for mask_filename in mask_filenames]

        # Resize FLAIR image and masks to target size
        # flair_image = flair_image.resize(self.img_size, Image.BICUBIC)
        # masks = [mask.resize(self.img_size, Image.NEAREST) for mask in masks]

        # # Convert images to tensors
        flair_image = transforms.ToTensor()(flair_image)
        masks = [transforms.ToTensor()(mask) for mask in masks]

        # Apply normalization to the FLAIR image
        # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # flair_image = normalize(flair_image)

        # Return FLAIR image and masks as a dictionary
        return {
            "flair_image": flair_image,
            "masks": torch.stack(masks)  # Stack the masks along the first dimension
        }

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def get_data_loaders_BRATS(flair_dir='dataset/2afc', masks_dir = '', batchsize=64, img_size=(64, 64), train_size=None, val_size=None, num_scales=4, num_levels=5, colour = False):
    # Initialize train and validation datasets
    train_dataset = BRATSDataset(flair_dir=flair_dir,masks_dir=masks_dir, dataset_type="train", img_size=img_size)
    val_dataset = BRATSDataset(flair_dir=flair_dir,masks_dir=masks_dir, dataset_type="val", img_size=img_size)

    # Determine sizes for training and validation datasets
    train_size = len(train_dataset) if not train_size else train_size
    val_size = len(val_dataset) if not val_size else val_size

    # Generate list of indices
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))

    # Create samplers for each split
    train_sampler = SubsetRandomSampler(train_indices[:train_size])
    val_sampler = SubsetRandomSampler(val_indices[:val_size])

    # Create data loaders using samplers
    train_loader = DataLoader(train_dataset, batch_size=batchsize, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=True)

    return train_loader, val_loader

import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_flair_and_masks(data_loader, num_samples=3):
    """
    Visualizes FLAIR images and their corresponding masks from the DataLoader.

    :param data_loader: The DataLoader to sample data from.
    :param num_samples: Number of samples to display.
    """
    # Iterate through the data loader
    for i, batch in enumerate(data_loader):
        # Get the flair images and masks from the batch
        flair_images = batch['flair_image']  # Shape: [batch_size, 1, height, width] for grayscale
        masks = batch['masks']  # Shape: [batch_size, num_masks, 1, height, width]

        # Display a few samples (num_samples)
        for j in range(min(num_samples, flair_images.size(0))):
            flair_image = flair_images[j].cpu().numpy()  # Convert to numpy for plotting
            mask = masks[j].squeeze(1).cpu().numpy()  # Remove the singleton dimension (1) for masks

            # print(f"Sample {j + 1} - FLAIR Image Shape: {flair_image.shape}")
            # print(f"Sample {j + 1} - Mask Shape: {mask.shape}")

            # Plot the flair image and masks
            fig, axes = plt.subplots(1, len(mask) + 1, figsize=(12, 5))

            # Show the flair image (convert to [0, 1] range for visualization)
            axes[0].imshow(flair_image[0], cmap='gray')  # Show as grayscale (only one channel)
            axes[0].set_title("FLAIR Image")
            axes[0].axis('off')

            # Show the masks
            for m in range(mask.shape[0]):
                axes[m + 1].imshow(mask[m], cmap='gray')  # Show each mask in grayscale
                axes[m + 1].set_title(f"Mask {m + 1}")
                axes[m + 1].axis('off')

            plt.tight_layout()
            plt.show()

        # If we have already displayed the desired number of samples, stop iterating
        if i >= num_samples // flair_images.size(0):
            break


current_dir = os.getcwd()  # Get current working directory
flair_path = os.path.join(current_dir, "data/flair_images")
ensemble_path = os.path.join(current_dir, "data/ensemble_predictions")

# Example usage
train_loader, val_loader = get_data_loaders_BRATS(
    flair_dir=flair_path,
    masks_dir=ensemble_path,
    batchsize=2, img_size=(64, 64)
)
visualize_flair_and_masks(train_loader, num_samples=3)




'''
Training setup for colour(2chrominance channels) metrics:
- model takes 2 channel images as input.
- apply color-based transformations and spatial transformations.
- Upthe image size is a lower resolution(64x64) then the y-channels
- currently using a sparsity distance of 2; as well as looking at cross-channel correlations between cr and cb

The code below configures and trains a SUPN model for the colour part of the metric.
'''

## adapted from paula right now

import os
import torch
print(torch.cuda.is_available())
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# from data_loader import get_data_loaders_BAAPS
from enum import Enum
import configparser
import wandb
import pdb

wandb.login()

class SupnBRATS:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Adjust according to your setup

        # Logging and data settings
        self.log_wandb = True  # Enable logging to wandb by default

        self.image_size = 64  # Image size set to 64 by default
        self.batch_size = 1  # Batch size set to 1 by default
        self.train_size = 5000  # Training data size set to 5000
        self.val_size = 200  # Validation data size set to 200
        self.data_loader_func = 'get_data_loaders_BRATS'  # Data loader function name (hardcoded)

        # Model settings
        self.local_connection_dist = [1]  # Local connection distance (hardcoded)
        self.num_levels = 5  # Number of levels in the model
        self.num_scales = 4  # Number of scales in the model
        self.num_pred_scales = 1  # Number of prediction scales
        self.dropout_rate = 0.0  # Dropout rate (no dropout)
        self.weight_decay = 0.0  # Weight decay set to 0 (no regularization)
        self.use_bias = False  # Do not use bias by default
        self.use_3D = False  # 3D operations are disabled by default
        self.max_ch = 1024  # Maximum number of channels in the model
        self.num_init_features = 32  # Number of initial features in the model
        # log diagonals needs to be 1 if 1 channel (mask 1,2,64,64 problem was caused by num_log_diag=2)
        self.num_log_diags = 1  # Number of log diagonals
        self.freeze_mean_decoder = True  # Freeze the mean decoder by default

        # File paths for saving and loading models
        self.supn_model_load_path = 'checkpoints/supn_model/colour1_lr0.0001_paramsnomean.pth'  # Path to load model
        self.supn_model_save_path = 'checkpoints/supn_model_ensebmle'  # Path to save the model

        # Optimizer settings
        self.optimizer_type = 'AdamW'  # Optimizer type set to AdamW

        os.makedirs(self.supn_model_save_path, exist_ok=True)

        self.train_schedules = [
            {
                'batch_size': 1,
                'learning_rate': 1e-3,
                'parameters': 'mean',
                'num_epochs': 1,
                'loss_type': 'mse_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-4,
                'parameters': 'mean',
                'num_epochs': 1, # 2
                'loss_type': 'mse_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'nomean',
                'num_epochs': 5, #10
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'dec',
                'num_epochs': 2, # 5
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'parameters': 'chol',
                'num_epochs': 2, # 5
                'loss_type': 'log_likelihood_loss'
            },
            {
                'batch_size': 1,
                'learning_rate': 1e-6,
                'parameters': 'chol',
                'num_epochs': 2, # 5
                'loss_type': 'log_likelihood_loss'
            }
        ]

        if self.log_wandb:
            wandb.init(project="SUPNBraTs")


        self.step = 0
        self.epoch = 0
        self.eval_freq = 10

        self.model = SUPNEncoderDecoder(
            in_channels=1,
            num_out_ch=1,
            num_init_features=self.num_init_features,
            num_log_diags=self.num_log_diags,
            num_local_connections=self.local_connection_dist,
            use_skip_connections=True,
            num_scales=self.num_scales,
            num_prediction_scales=self.num_pred_scales,
            max_dropout=self.dropout_rate,
            use_bias=self.use_bias,
            use_3D=self.use_3D,
            max_ch=self.max_ch
        ).to(self.device)

        self.load_data()

        if self.supn_model_load_path and os.path.exists(self.supn_model_load_path):
            self.model.load_state_dict(torch.load(self.supn_model_load_path, map_location=self.device))


    def load_data(self):
        loader_func = eval(self.data_loader_func)
        self.train_dataloader, self.val_dataloader = loader_func(
            flair_path,
            ensemble_path,
            self.batch_size,
            (self.image_size, self.image_size),
        )

    # def __init__(self, flair_dir='dataset/2afc', masks_dir='', dataset_type="train", img_size=(256, 256), num_masks=5):


    def freeze_parameters(self, mode):
        if mode == 'mean':
            self._unfreeze_mean_decoder()
        if mode == 'nomean':
            self._freeze_mean_decoder()
        elif mode == 'dec':
            self._freeze_decoder()
        elif mode == 'chol':
            self._freeze_precision_decoder()
        elif mode == 'all':
            self._unfreeze_all()

    def _unfreeze_mean_decoder(self): # freezes everything but mean decoder, and encoder
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.encoder_decoder.supn_decoder.parameters():
            param.requires_grad = False

    def _freeze_mean_decoder(self): # freezes the mean decoder
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.encoder_decoder.mean_decoder.parameters():
            param.requires_grad = False

    def _freeze_decoder(self): # freezes everything(encodser,bottleneck) but the mean and supn decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder_decoder.mean_decoder.parameters():
            param.requires_grad = True
        for param in self.model.encoder_decoder.supn_decoder.parameters():
            param.requires_grad = True

    def _freeze_precision_decoder(self): # freezes everything but the choleski decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder_decoder.supn_decoder.parameters():
            param.requires_grad = True

    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True


    def save_model(self, schedule):
        save_file = os.path.join(self.supn_model_save_path, f"{schedule}.pth")
        torch.save(self.model.state_dict(), save_file)
        print(f"Model saved to {save_file}")


    def get_losses_one_scale_supn(self, supn_data, target, level=0, only_mean=False):
        # assert isinstance(supn_data, SUPN)
        weighting_factor = 1.0 / (level + 1) * 5 # penelty based on level of transformation
        #print("Does supn_data.mean require gradients?", supn_data.mean.requires_grad)
        mse_loss_fn = nn.MSELoss(reduction='mean')
        # mse_loss = mse_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))
        # print(f"supn_data.mean shape: {supn_data.mean.shape}")
        # print(f"supn_data.mean squeeze shape: {supn_data.mean.squeeze(0).shape}")
        # print(f"target shape: {target.shape}")
        mse_loss = mse_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))
        mse_loss = mse_loss * weighting_factor

        # trying out hybrid loss
        bce_loss_fn = nn.BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(supn_data.supn_data.mean.squeeze(0), target.to(self.device))

        combined_loss = 0.5*mse_loss + 0.5*bce_loss

        # before instead of combined it was just mse
  
        assert isinstance(mse_loss, torch.Tensor)

        if only_mean:
            return {'mse_loss': combined_loss}
        else:
            log_prob = -supn_data.log_prob(target.to(self.device)) * weighting_factor
            return {'mse_loss': combined_loss, 'log_likelihood_loss': log_prob.mean()}


    def run_model(self, image):
        image = image.to(self.device)
        return self.model(image)


    def run_epoch(self, loss_type):
        self.model.train()
        gradient_accumulation_steps = 10
        self.optimizer.zero_grad()

        accumulated_loss = 0.0  # logging

        for step, batch in enumerate(self.train_dataloader):
            loss = self.run_batch(batch, loss_type)[loss_type]
            loss = loss / gradient_accumulation_steps  #Normalize loss
            loss.backward()
            accumulated_loss += loss.item()  #logging

            if (step + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                #print('accumulated',accumulated_loss)

                if self.log_wandb:
                    wandb.log({
                        f'{loss_type}': accumulated_loss,
                        'Step': self.step
                    })


                accumulated_loss = 0.0


    def run_batch(self, batch, loss_type='log_likelihood_loss'):
        loss_dict = {
            'mse_loss': torch.tensor(0.0, device=self.device),
            'log_likelihood_loss': torch.tensor(0.0, device=self.device)
        }

        flair_images = batch["flair_image"]  # Flair images for the batch
        masks = batch["masks"]  # Masks are the ensemble predictions (e.g., 3 masks per flair image)

        # print(f"Flair images shape: {flair_images.shape}")
        # print(f"Masks shape: {masks.shape}")

        for idx, flair_image in enumerate(flair_images):  # Loop through each flair image
            flair_image = flair_image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

            # print(f"Flair image shape: {flair_image.shape}")
            # print(f"Flair image unsqueeze shape: {flair_image.unsqueeze(0).shape}")
            # print(f"Masks shape: {masks.shape}")

            for mask_set in masks:  # Loop through each set of masks for this particular flair image
                mask = mask_set[idx].to(self.device)  # Get the corresponding mask for this flair image

                # Run the model on the flair image
                supn_outputs = self.run_model(flair_image)  # Generate model output for this flair image

                # Calculate the loss between the model output and the mask
                only_mean = loss_type != 'log_likelihood_loss'
                leveled_loss_dict = self.get_losses_one_scale_supn(
                    supn_outputs[0], mask, level=0, only_mean=only_mean
                )

                # Accumulate losses
                for loss_name, loss_value in leveled_loss_dict.items():
                    loss_dict[loss_name] += loss_value

        return loss_dict





    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss_dict = self.run_batch(batch, loss_type='log_likelihood_loss')
                if self.log_wandb:
                    if 'mse_loss' in loss_dict:
                        wandb.log({'Validation/mse_loss': loss_dict.get('mse_loss').item() , 'Step': self.step})
                    if 'log_likelihood_loss' in loss_dict:
                        wandb.log({'Validation/log_likelihood_loss': loss_dict.get('log_likelihood_loss'), 'Step': self.step})

    def set_optimizer(self, learning_rate):
        if self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")


    def load_model(self):
        self.model.load_state_dict(torch.load(self.supn_model_load_path))


    def train(self):
        for schedule_idx, schedule in enumerate(self.train_schedules):
            self.set_optimizer(schedule['learning_rate'])
            self.freeze_parameters(schedule['parameters'])
            print(f"Training with parameters: {schedule['parameters']}")
            for epoch in range(schedule['num_epochs']):
                self.run_epoch(schedule['loss_type'])
                # self.validate()
                print(epoch)
                stage_name = f"brats{schedule_idx}_lr{schedule['learning_rate']}_params{schedule['parameters']}"
                self.save_model(stage_name)


if __name__ == '__main__':
    trainer = SupnBRATS()
    trainer.train()


