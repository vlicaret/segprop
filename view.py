# import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import torch
# matplotlib.use('TkAgg')


def imshow(img, figure=1):
    if str(img.dtype).find('torch') > -1:
        img = img.cpu().numpy()
    plt.figure(figure)
    plt.imshow(img)
    plt.show(block=False)
    return 0


def merge(img1, img2, value=0.5):
    img = img1 * value + img2 * (1 - value)
    return img.round().type(torch.uint8)


def imsave(img, name):
    if str(img.dtype).find('torch') > -1:
        img = img.cpu().numpy()
    io.imsave(name, img)
