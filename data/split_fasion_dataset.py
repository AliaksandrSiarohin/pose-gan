import os
import numpy as np
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Split faison dataset so it has more or less same format as market")
parser.add_argument("--input_dir", default='fasion', help='Fasion is folder with 2 subfolders (MAN, WOMEN)')
parser.add_argument("--output_dir", default='fasion-dataset', help="Result directory")
parser.add_argument("--annotations_file_train", default='fasion-annotation-train.csv', help='File with train annotations')
parser.add_argument("--annotations_file_test", default='fasion-annotation-test.csv', help='File with train annotations')

args = parser.parse_args()

output_folder_test = os.path.join(args.output_dir, 'test')
output_folder_train = os.path.join(args.output_dir, 'train')

def get_id(image_path):
    path_names = image_path.split(r'/')
    path_names[2] = path_names[2].replace('_', '')
    path_names[3] = path_names[3].replace('_', '')
    path_names[4] = path_names[4].split('_')[0]
    return ''.join(path_names)

def get_pose_name(image_path):
    path_names = image_path.split(r'/')
    return ''.join(path_names[4].split('_')[1:])

images = []
for root, _, files in os.walk('fasion'):
    images += [os.path.join(root, name) for name in files]

train_images_df = pd.read_csv(args.annotations_file_train, sep=':')
test_images_df = pd.read_csv(args.annotations_file_test, sep=':')

train_images = set(train_images_df['name'])
test_images = set(test_images_df['name'])


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.exists(output_folder_train):
    os.makedirs(output_folder_train)

if not os.path.exists(output_folder_test):
    os.makedirs(output_folder_test)

for image_path in images:
    id = get_id(image_path)
    pose = get_pose_name(image_path)
    result_name = id + '_' + pose
    if result_name in test_images:
        shutil.copy(image_path, dst=os.path.join(output_folder_test, result_name))
    elif result_name in train_images:
        shutil.copy(image_path, dst=os.path.join(output_folder_train, result_name))
    
