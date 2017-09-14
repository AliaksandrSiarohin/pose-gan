import os
import re
import sys

import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
from joblib import Parallel, delayed


from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
import cv2

from keras.models import load_model
import skimage.transform as st
import pandas as pd
from tqdm import tqdm

model = load_model('pose_estimator.h5')
test_folder = sys.argv[1]
output_path = sys.argv[2]
thre1 = 0.1
boxsize = 368
scale_search = [0.5, 1, 1.5, 2]

if os.path.exists(output_path):
    processed_names = set(pd.read_csv(output_path, sep=':')['name'])
    result_file = open(output_path, 'a')
else:
    result_file = open(output_path, 'w')
    processed_names = set()
    print >>result_file, 'name: keypoints_x: keypoints_y'


for image_name in tqdm(os.listdir(test_folder)):
	if image_name in processed_names:
	    continue
	oriImg = cv2.imread(os.path.join(test_folder, image_name)) # B,G,R order

	multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]

	heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
	paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

	for m in range(len(multiplier)):
	    scale = multiplier[m]

	    imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
	    imageToTest_padded = np.float32(imageToTest[np.newaxis,:,:,:])/256 - 0.5
	    
	    output1,output2 = model.predict(imageToTest_padded)

	    heatmap = st.resize(output2[0], oriImg.shape[:2], preserve_range = True, order = 1)    
	    paf = st.resize(output1[0], oriImg.shape[:2], preserve_range = True, order = 1)     
	    heatmap_avg += heatmap
	    paf_avg += paf


	heatmap_avg /= len(multiplier)
	paf_avg /= len(multiplier)
	all_peaks = []
	peak_counter = 0

	for part in range(18):
	    map_ori = heatmap_avg[:,:,part]
	    map = gaussian_filter(map_ori, sigma=3)
	    
	    map_left = np.zeros(map.shape)
	    map_left[1:,:] = map[:-1,:]
	    map_right = np.zeros(map.shape)
	    map_right[:-1,:] = map[1:,:]
	    map_up = np.zeros(map.shape)
	    map_up[:,1:] = map[:,:-1]
	    map_down = np.zeros(map.shape)
	    map_down[:,:-1] = map[:,1:]
	    
	    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > thre1))
	    
	    peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])
	    
	    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
	    id = range(peak_counter, peak_counter + len(peaks))
	    peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

	    all_peaks.append(peaks_with_score_and_id)
	    peak_counter += len(peaks)

	none_value = -1
	x_values = []
	y_values = []

	for i in range(18):
	    if len(all_peaks[i]) != 0:
		x_values.append(all_peaks[i][0][0])
		y_values.append(all_peaks[i][0][1])
	    else:
		x_values.append(none_value)
		y_values.append(none_value)

	print >>result_file, "%s: %s: %s" % (image_name, str(x_values), str(y_values))
	result_file.flush()
