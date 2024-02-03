import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from model.commons import Squeeze, Reshape
from utils.conf import Configuration


class _PreActivatedResBlock(nn.Module):
    def __init__(self, conf: Configuration, in_channels, out_channels, dilation, first=False, last=False) -> None:
        super(_PreActivatedResBlock, self).__init__()

        kernel_size = conf.getHP('kernel_size')
        dim_series = conf.getHP('dim_series')
        padding = int(kernel_size / 2) * dilation

        if first:
            self.__first_block = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=True)
            in_channels = out_channels
        else:
            self.__first_block = nn.Identity()

        self.__residual_link = nn.Sequential(nn.LayerNorm(dim_series, elementwise_affine=True),
                                             nn.ReLU(),
                                             nn.Conv1d(
                                                 in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=True),

                                             nn.LayerNorm(dim_series),
                                             nn.ReLU(),
                                             nn.Conv1d(
                                                 in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=True)
                                             )

        if in_channels != out_channels:
            self.__identity_link = nn.Conv1d(
                in_channels, out_channels, 1, bias=True)
        else:
            self.__identity_link = nn.Identity()

        if last:
            self.__after_addition = nn.Sequential(nn.LayerNorm(dim_series),
                                                  nn.ReLU())
        else:
            self.__after_addition = nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        input = self.__first_block(input)

        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)


class _ResNet(nn.Module):
    def __init__(self, conf: Configuration):
        super(_ResNet, self).__init__()

        num_resblock = conf.getHP('num_en_resblock')
        assert num_resblock > 1 and 2 ** (num_resblock +
                                          1) <= conf.getHP('dim_series') + 1

        inner_channels = conf.getHP('num_en_channels')
        out_channels = conf.getHP('dim_en_latent')

        layers = [_PreActivatedResBlock(
            conf, 1, inner_channels, conf.getDilation(1), first=True)]
        layers += [_PreActivatedResBlock(conf, inner_channels, inner_channels,
                                         conf.getDilation(depth=depth)) for depth in range(2, num_resblock)]
        layers += [_PreActivatedResBlock(conf, inner_channels,
                                         out_channels, conf.getDilation(num_resblock), last=True)]

        self.__model = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)


class TSEncoder(nn.Module):
    def __init__(self, conf: Configuration):
        ''' Initialize a TS2Vec Model

        Args:
            in_channels (int): num_en_channels, the number of encoders' channel; also called num_channels
            out_channels (int): dim_en_latent, the dimension of latent feature of the encoder; also called dim_latent
            dim_series (int): the length of the original series
            dim_embedding (int): the dimension of the embedding
            kernel_size (int): the kernel size of convolution layer
            num_resblock (int): The number of resblocks.
        '''

        super(TSEncoder, self).__init__()

        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_en_channels')
        dim_latent = conf.getHP('dim_en_latent')

        self.__model = nn.Sequential(
            _ResNet(conf),
            nn.AdaptiveAvgPool1d(1),
            Squeeze(),

            nn.Linear(num_channels, dim_latent),
            nn.Tanh(),

            nn.Linear(dim_latent, dim_embedding, bias=False),
            nn.LayerNorm(dim_embedding, elementwise_affine=False)
        )

        self.__model.to(conf.getHP('device'))

    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)


class TSDecoder(nn.Module):
    def __init__(self, conf: Configuration) -> None:
        super(TSDecoder, self).__init__()

        dim_series = conf.getHP('dim_series')
        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_de_channels')
        dim_latent = conf.getHP('dim_de_latent')

        self.__model = nn.Sequential(Reshape([-1, 1, dim_embedding]),
                                     nn.Linear(dim_embedding, dim_series),
                                     nn.Tanh(),

                                     _ResNet(conf),
                                     nn.AdaptiveMaxPool1d(1),
                                     Reshape([-1, 1, num_channels]),

                                     nn.Linear(num_channels, dim_latent),
                                     nn.Tanh(),

                                     nn.Linear(
                                         dim_latent, dim_series, bias=False),
                                     nn.LayerNorm(dim_series, elementwise_affine=False))

        self.__model.to(conf.getHP('device'))

    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)
