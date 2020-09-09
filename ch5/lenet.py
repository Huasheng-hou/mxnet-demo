import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon.nn import Sequential, Conv2D, Dense, MaxPool2D

net = Sequential()
net.add(Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        MaxPool2D(pool_size=2, strides=2),
        Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        MaxPool2D(pool_size=2, strides=2),
        Dense(120, activation='sigmoid'),
        Dense(84, activation='sigmoid'),
        Dense(10))

batch_size = 256
train_iter, test_iter = d2l.load_data_mnist(batch_size=batch_size)

lr, num_epochs = 0.9, 5
ctx = d2l.try_gpu()

net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())

X = nd.random.uniform(shape=(1, 1, 28, 28))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
