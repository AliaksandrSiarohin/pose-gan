import matplotlib
matplotlib.use('Agg')
import pylab as plt

import os
#import tensorflow as tf
from keras.models import load_model
import skimage.transform
from skimage.io import imread
from skimage import img_as_ubyte
#from keras import backend as K
import numpy as np
import pandas as pd
import json
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

model = load_model('/data2/aliaksandr/pose-gan/pose_estimator.h5')
# model = pd.read_csv('pose_anotations.csv', sep=':')
# print (model.head())



def plot_images(imgs, model, name, nrows = 6, ncols = 6):
    plt.figure(figsize = (20, 20))
    for i in range(nrows * ncols):
        if imgs[i].startswith('-1'):
            continue
        img = imread(imgs[i])
        print (imgs[i])        
        plt.subplot(nrows, ncols, i + 1)

        
        points = model.predict(preprocess_input(np.expand_dims(resize(img, (299, 299), preserve_range = True), 0)))
        

        keypoints_x = points[0][::2]
        keypoints_y = points[0][1::2]     
        
        plt.imshow(img)
        plt.scatter(keypoints_x, keypoints_y)
    plt.savefig(name)

    
dir = '/data2/aliaksandr/market-dataset/bounding_box_train/'
imgs = [os.path.join(dir, name) for name in os.listdir(dir)]
plot_images(imgs, model, 'market_train.png')

dir = '/data2/aliaksandr/pose-gan/images'
imgs = [os.path.join(dir, name) for name in os.listdir(dir)]
plot_images(imgs, model, 'market_generated.png')
