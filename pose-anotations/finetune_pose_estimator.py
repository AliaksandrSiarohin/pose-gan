#from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import SGD
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import os
from tqdm import tqdm
import pandas as pd
#from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
IMAGES_TRAIN = '/data2/aliaksandr/market-dataset/bounding_box_train/'
ANOTATIONS_TRAIN = '/data2/aliaksandr/pose-gan/pose_anotations.csv'
MODEL_PATH = '/data2/aliaksandr/pose-gan/pose_estimator.h5'
EPOCHS = 40
IMAGE_SIZE = (299, 299)

def create_model(input_tensor):
    # pre-built and pre-trained deep learning VGG16 model
    x = InceptionV3(weights='imagenet', include_top=False, input_tensor = input_tensor) (input_tensor)
    
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(32, name='predictions')(x)
    
#     x = Flatten(name='flatten')(x)
#     x = Dense(4096, activation='relu', name='fc1')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(4096, activation='relu', name='fc2')(x)
#     x = BatchNormalization()(x)
#     x = Dense(32, name='predictions')(x)
    
    return Model(inputs = input_tensor, outputs = x)

model = create_model(Input((IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
model.summary()

model.compile(optimizer=SGD(0.0001, momentum=0.9, decay = 1e-6, nesterov=True), loss='mean_squared_error')

def read_dataset(dir, anotations):
    import json
    
    x_train = np.empty((len(df), IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype='float32')
    y_train = np.empty((len(df), 32))

    print ("Reading dataset...")
    for index, row in tqdm(anotations.iterrows()):
        x_train[index] = resize(imread((os.path.join(dir, row['name']))), 
                                (IMAGE_SIZE[0], IMAGE_SIZE[1]), preserve_range = True)
        y_train[index, ::2] = json.loads(row['keypoints_x'])
        y_train[index, 1::2] = json.loads(row['keypoints_y'])
    
    return preprocess_input(x_train), y_train


df = pd.read_csv(ANOTATIONS_TRAIN, sep=':')
#df_train, df_validation = train_test_split(df, test_size = 0.1, random_state = 0)
x_train, y_train = read_dataset(IMAGES_TRAIN, df)
#x_val, y_val = read_dataset(IMAGES_TRAIN, df_validation)

model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2)

model.save(MODEL_PATH)
