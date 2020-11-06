from PIL import Image
import numpy as np
import d2lzh as d2l
import os
from random import randint
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata
from mxnet.gluon.nn import Sequential, HybridSequential, Conv2D, Flatten, Conv2DTranspose, Dense, MaxPool2D

dirs = os.listdir('background')
imgs = []
for path in dirs:
    img = Image.open('background/' + path)
    imgs.append(img.convert('L'))


def getPatch(imgs=None):
    length = len(imgs)
    img = imgs[randint(0, length - 1)]
    x = randint(0, img.size[0] - 29)
    y = randint(0, img.size[1] - 29)
    return img.crop((x, y, x+28, y+28))


def blendData(data_iter=None, imgs=None):

    features = []
    labels = []

    i = 0
    for X, y in data_iter:
        digit = X * 255
        digit = digit[0][0].asnumpy()
        bg = np.asarray(getPatch(imgs=imgs))
        res = np.uint16(digit) + np.uint16(bg)
        res[res > 255] = 255
        digit[digit > 0] = 1
        res = nd.array(res).reshape(1, 28, 28)
        digit = nd.array(digit).reshape(1, 28, 28)
        features.append(res)
        labels.append(digit.astype(int))
        # i += 1
        # if i == 256 * 10:
        #     break

    dataset = gdata.ArrayDataset(features, labels)
    d_iter = gdata.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return d_iter


batch_size = 256

train_iter, test_iter = d2l.load_data_mnist(batch_size=1)

train_iter = blendData(train_iter, imgs)
test_iter = blendData(test_iter, imgs)

num_classes = 2

net = HybridSequential()
net.add(Conv2D(channels=3, kernel_size=5, activation='sigmoid'),
        MaxPool2D(pool_size=2, strides=2),
        Conv2D(channels=8, kernel_size=5, activation='sigmoid'),
        MaxPool2D(pool_size=2, strides=2),
        Conv2D(channels=120, kernel_size=4, activation='sigmoid'),
        Conv2D(channels=84, kernel_size=1, activation='sigmoid'),
        Conv2D(channels=10, kernel_size=1),
        Conv2DTranspose(num_classes, kernel_size=56, padding=14, strides=28, activation='sigmoid'))

lr, num_epochs = 0.9, 10
ctx = d2l.try_gpu()

net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
