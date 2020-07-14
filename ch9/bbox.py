import d2lzh as d2l
from mxnet import image

d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
# d2l.plt.imshow(img)

dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

fig = d2l.plt.imshow(img)
fig.axes.add_patch(d2l.bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(d2l.bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()
