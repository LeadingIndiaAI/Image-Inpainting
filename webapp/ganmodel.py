import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
import base64
import yaml
import cv2
import numpy as np
from PIL import Image

from flask import request
from flask import jsonify
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

input_dim = 5
output_dim = 5
#kernel_size

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='lrelu', pad_type='zeros', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)

        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias, padding_mode=pad_type)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)

class GlobalLocalAttention(nn.Module):
    def __init__(self, in_dim, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=True):
        super(GlobalLocalAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.chanel_in = in_dim
        self.feature_attention = GlobalAttention(in_dim)
        self.patch_attention = GlobalAttention(in_dim)

    def forward(self, f, b, mask=None):
        # get shapes
        raw_int_fs = list(f.size())  # b*c*h*w
        raw_int_bs = list(b.size())  # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate * self.stride,
                                               self.rate * self.stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.rate, mode='nearest')
        int_fs = list(f.size())  # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        b_groups = torch.split(b, 1, dim=0)  # split tensors along the batch dimension
        m_groups = torch.split(mask, 1, dim=0)  # split tensors along the batch dimension
        mask = m_groups[0]
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        fw = extract_image_patches(f, ksizes=[self.ksize, self.ksize],
                                   strides=[self.stride, self.stride],
                                   rates=[1, 1],
                                   padding='same')
        # w shape: [N, C, k, k, L]
        fw = fw.view(int_fs[0], int_fs[1], self.ksize, self.ksize, -1)
        fw = fw.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        fw_groups = torch.split(fw, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            down_rate = mask.shape[2] // int_bs[2]
            mask = F.interpolate(mask, scale_factor=1. / (down_rate), mode='nearest')
            #mask = mask.cuda()
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]
        m = m[0]  # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        k = self.fuse_k
        scale = self.softmax_scale  # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        #fuse_weight = fuse_weight.cuda()

        for xi, bi, fi, wi, raw_wi in zip(f_groups, b_groups, fw_groups, w_groups, raw_w_groups):
            escape_NaN = torch.FloatTensor([1e-4])
            #if self.use_cuda:
             #   escape_NaN = escape_NaN.cuda()

            # Selecting patches
            fi = fi[0]
            wi = wi[0]
            # Patch Level Global Attention
            m_batchsize_p, C_p, width_p, height_p = fi.size()
            final_pruning_p = self.patch_attention(fi, wi, m)
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(final_pruning_p, 2), axis=[1, 2, 3], keepdim=True)),
                               escape_NaN)
            wi_normed = final_pruning_p / max_wi

            # # Global Attention
            m_batchsize, C, width, height = xi.size()  # B, C, H, W
            final_pruning = self.feature_attention(xi, bi, mask)
            final_pruning = same_padding(final_pruning, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(final_pruning, wi_normed, stride=1)  # [1, L, H, W]
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # # softmax to match
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm
            # deconv for patch pasting
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        # y = F.pad(y, [0, 1, 0, 1])  # here may need conv_transpose same padding
        y.contiguous().view(raw_int_fs)

        return y

class GlobalAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #
        self.rate = 1
        self.gamma = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad = True)
        self.gamma.requires_grad = True

    def forward(self, a, b, c):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                c : B * 1 * W * H
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = a.size()  # B, C, H, W
        down_rate = int(c.size(2)//width)
        c = F.interpolate(c, scale_factor=1./down_rate*self.rate, mode='nearest')
        proj_query = self.query_conv(a).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B, C, N -> B N C
        proj_key = self.key_conv(b).view(m_batchsize, -1, width * height)  # B, C, N
        feature_similarity = torch.bmm(proj_query, proj_key)  # B, N, N

        mask = c.view(m_batchsize, -1, width * height)  # B, C, N
        mask = mask.repeat(1, height * width, 1).permute(0, 2, 1)  # B, 1, H, W -> B, C, H, W // B

        feature_pruning = feature_similarity * mask
        attention = self.softmax(feature_pruning)  # B, N, C
        feature_pruning = torch.bmm(self.value_conv(a).view(m_batchsize, -1, width * height),
                                    attention.permute(0, 2, 1))  # -. B, C, N
        out = feature_pruning.view(m_batchsize, C, width, height)  # B, C, H, W
        out = self.gamma * a*c + (1.0- c) * out
        return out

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def tensor_img_to_npimg(tensor_img):
    """
    Turn a tensor image with shape CxHxW to a numpy array image with shape HxWxC
    :param tensor_img:
    :return: a numpy array image with shape HxWxC
    """
    if not (torch.is_tensor(tensor_img) and tensor_img.ndimension() == 3):
        raise NotImplementedError("Not supported tensor image. Only tensors with dimension CxHxW are supported.")
    npimg = np.transpose(tensor_img.numpy(), (1, 2, 0))
    npimg = npimg.squeeze()
    assert isinstance(npimg, np.ndarray) and (npimg.ndim in {2, 3})
    return npimg


# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])
    return torch.stack(patches, dim=0)


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def deprocess(img):
    img = img.add_(1).div_(2)
    return img

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_dim = 5
        self.cnum = 32

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2 = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum):
        super(CoarseGenerator, self).__init__()

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)
        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu')
        self.contextul_attention = GlobalAttention(in_dim=128)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum // 2, 3, 3, 1, 1, activation='none')

    def forward(self, xin, mask):
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        # conv branch
        xnow = torch.cat([xin, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1



class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum):
        super(FineGenerator, self).__init__()

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1)
        # cnum*4 x 64 x 64

        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu')
        self.contextul_attention = GlobalLocalAttention(in_dim=128, ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)

        self.allconv17 = gen_conv(cnum // 2, 3, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2


def result(image_inp, mask_inp):
    image_shape = [256, 256, 3]
    
    r_seed = 2019 #random.randint(1, 10000)

    with torch.no_grad():   # enter no grad context
    #if is_image_file(image_inp):
        #if is_image_file(mask_inp):
        x = default_loader(image_inp)
        x = transforms.Resize(image_shape[:-1])(x)
        x = transforms.ToTensor()(x)
        x = normalize(x)
        mask = default_loader(mask_inp)
        mask = transforms.Resize(image_shape[:-1])(mask)
        mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
        x = x * (1. - mask)
        x = x.unsqueeze(dim=0)
        x_raw = x
        mask = mask.unsqueeze(dim=0)
          
        #else:
            #raise TypeError("{} is not an image file.".format(mask_inp))
                       
                
# Define the trainer
    netG_one = Generator()
# Resume weight
    g_checkpoint = torch.load(APP_ROOT+'/gen1.pt', map_location=torch.device('cpu'))
    netG_one.load_state_dict(g_checkpoint, strict=False)
# model_iteration = int(last_model_name[-11:-3])
    print("Model1 Resumed")

    x1, x2 = netG_one(x, mask)
    inpainted_result1 = x2 * mask + x * (1. - mask)

    #print(inpainted_result1)
    print(inpainted_result1.shape)

    vutils.save_image(inpainted_result1, APP_ROOT+'/static/img/output1.png', padding=0, normalize=True)
    print('output1 saved')

    #if cuda:
     #   netG_one = nn.parallel.DataParallel(netG_one, device_ids=device_ids)
      #  x = x.cuda()
       # mask = mask.cuda()

# Define the trainer
    netG_two = Generator()
    # Resume weight
    g_checkpoint = torch.load(APP_ROOT+'/gen2.pt', map_location=torch.device('cpu'))
    netG_two.load_state_dict(g_checkpoint, strict=False)
    # model_iteration = int(last_model_name[-11:-3])
    print("Model2 Resumed")

    x1, x2 = netG_two(x, mask)
    inpainted_result2 = x2 * mask + x * (1. - mask)

    print(inpainted_result2.shape)

    vutils.save_image(inpainted_result2, APP_ROOT+'/static/img/output2.png', padding=0, normalize=True)
    print('output2 saved')

    #if cuda:
     #   netG_two = nn.parallel.DataParallel(netG_two, device_ids=device_ids)
      #  x = x.cuda()
       # mask = mask.cuda()
    

    allfiles=[APP_ROOT+'/static/img/output1.png',APP_ROOT+'/static/img/output2.png']
    imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]

    # Assuming all images are the same size, get dimensions of first image
    w,h=Image.open(imlist[0]).size
    N=len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr=np.zeros((h,w,3),np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr=np.array(Image.open(im),dtype=np.float)
        arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr),dtype=np.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    #out.save('/content/output.png')
    #out.show()

    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr),dtype=np.uint8)

    #print(arr3)
    # Generate, save and preview final image


    #cat_img = torch.stack((TF.to_tensor(img1), TF.to_tensor(img2)), dim=0)

    cat_one = TF.to_tensor(arr)

    #print(cat_img)
    op_name = image_inp.split('/')[-1]
    vutils.save_image(cat_one, APP_ROOT+'/static/uploads/output/'+op_name, padding=0, normalize=True)
#Ensemble
"""
    class MyEnsemble(nn.Module):
        def __init__(self, modelA, modelB):
            super(MyEnsemble, self).__init__()
            self.modelA = netG_one
            self.modelB = netG_two
            self.classifier = nn.Linear(4, 2)
        
        def forward(self, x1, x2):
            x1 = self.modelA(x1)
            x2 = self.modelB(x2)
            x = torch.cat((x1, x2), dim=1)
            x = self.classifier(F.relu(x))
            return x

    # Inference
    model = MyEnsemble(netG_one, netG_two)
    x1, x2 = model(x, mask)
    inpainted_result = x2 * mask + x * (1. - mask)

    vutils.save_image(inpainted_result, '/content/output.png', padding=0, normalize=True)
    print('output saved')

    #except Exception as e:  # for unexpected error logging
     #   print("Error: {}".format(e))
      #  raise e
"""

if __name__ == '__main__':
    app.run(debug=True)