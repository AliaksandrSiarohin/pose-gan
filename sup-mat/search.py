### Script for retriving images from paper in large datasets, script uses vgg conv5 descriptors for comparizon.
### usage: python search.py /path/to/dataset/folder /path/to/images/from/paper/folder

from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
import os
from keras.applications import vgg19
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import glob
from skimage.io import imread



def compute_descriptor(batch):
    if not hasattr(compute_descriptor, "vgg_model"):
        compute_descriptor.vgg_model = vgg19.VGG19(include_top=False, input_shape=(None, None, 3))
    descriptors = []
    for shape in [(224, 224)]:
        batch = resize(batch, shape, preserve_range=True)
        descriptor = vgg19.preprocess_input(np.expand_dims(batch, axis=0))
        descriptors.append(compute_descriptor.vgg_model.predict(descriptor))
    return descriptors


def main(image_folder, search_image_folder):
	image_names = os.listdir(search_image_folder)
	search_images = [imread(os.path.join(search_image_folder, search_image)) for search_image in image_names]
	descriptors = [compute_descriptor(img) for img in search_images]

	min_val = [1e100] * len(search_images)
	at_min = [None] * len(search_images)
	for img_name in tqdm(os.listdir(image_folder)):
		img = imread(os.path.join(image_folder,img_name))
		dst = compute_descriptor(img)

		for i, descriptor in enumerate(descriptors):
			val = np.sum([np.mean((d1-d2)**2) for d1, d2 in zip(descriptor, dst)])
			if min_val[i] > val:
				min_val[i] = val
				at_min[i] = img_name

	print (sorted(zip(image_names, at_min, min_val)))

if __name__ == "__main__":
	import sys
	main(sys.argv[1], sys.argv[2])
