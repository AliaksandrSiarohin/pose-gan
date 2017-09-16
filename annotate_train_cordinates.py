import pose_utils
import os
import sys
import numpy as np

from keras.models import load_model
import skimage.transform as st
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

model = load_model('cao-hpe/pose_estimator.h5')

input_folder = sys.argv[1]
output_path = sys.argv[2]

threshold = 0.1
boxsize = 368
scale_search = [0.5, 1, 1.5, 2]

if os.path.exists(output_path):
    processed_names = set(pd.read_csv(output_path, sep=':')['name'])
    result_file = open(output_path, 'a')
else:
    result_file = open(output_path, 'w')
    processed_names = set()
    print >> result_file, 'name:keypoints_y:keypoints_x'

for image_name in tqdm(os.listdir(input_folder)):
    if image_name in processed_names:
        continue

    oriImg = imread(os.path.join(input_folder, image_name))[:, :, ::-1]  # B,G,R order

    multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        new_size = (np.array(oriImg.shape[:2]) * scale).astype(np.int32)
        imageToTest = resize(oriImg, new_size, order=3, preserve_range=True)
        imageToTest_padded = imageToTest[np.newaxis, :, :, :]/255 - 0.5

        output1, output2 = model.predict(imageToTest_padded)

        heatmap = st.resize(output2[0], oriImg.shape[:2], preserve_range=True, order=1)
        paf = st.resize(output1[0], oriImg.shape[:2], preserve_range=True, order=1)
        heatmap_avg += heatmap
        paf_avg += paf

    heatmap_avg /= len(multiplier)

    np.save(image_name, heatmap_avg)
    pose_cords = pose_utils.map_to_cord(heatmap_avg, threshold=threshold)

    print >> result_file, "%s: %s: %s" % (image_name, str(list(pose_cords[:, 0])), str(list(pose_cords[:, 1])))
    result_file.flush()
