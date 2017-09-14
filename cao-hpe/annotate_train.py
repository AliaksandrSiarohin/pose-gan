from keras.models import Sequential, Model, model_from_json, load_model, Input
import numpy as np
import sys
import os
from tqdm import tqdm
from skimage.io import imread

dir_in = sys.argv[1]
dir_out = sys.argv[2]

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# model = model_from_json(open('pose_estimator_tensorflow.json').read())
model = load_model('pose_estimator.h5')
# a = Input((None, None, 3))
# model = Model(inputs = a, outputs = model(a))

def preprocess_input(a):
    return (a[..., ::-1] / 256.0) - 0.5

for name in tqdm(os.listdir(dir_in)):
    img = imread(os.path.join(dir_in, name))
    inp = preprocess_input(np.expand_dims(img, 0))
    _, out = model.predict(inp)
    np.save(os.path.join(dir_out, name), np.squeeze(out, axis=0))