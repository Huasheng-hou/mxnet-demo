import d2lzh as d2l
import mxnet as mx
from mxnet import gluon, init, nd
from mxnet.gluon import block, nn
from mxnet.gluon.nn import Sequential, Block, Dense, Conv2D, MaxPool2D


def vgg_block(num_convs, num_channels):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(Conv2D(num_channels, kernel_size=3,
                       padding=1, activation='relu'))
    blk.add(MaxPool2D(pool_size=2, strides=2))
    return blk


def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层部分
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5))
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5))
    net.add(nn.Dense(10))
    return net




