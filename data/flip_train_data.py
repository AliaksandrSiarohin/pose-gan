from skimage.io import imread, imsave
import os
import sys

dataset = 'market' if len(sys.argv) == 1 else sys.argv[1]

dir = dataset + "-dataset/train"
for img_name in os.listdir(dir):
    img = imread(os.path.join(dir, img_name))
    img = img[:, ::-1]
    imsave(os.path.join(dir, img_name.replace('.jpg', 'r.jpg')), img)
