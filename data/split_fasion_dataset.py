import os
import numpy as np
import shutil



#Fasion is folder with 2 subfolders (MAN, WOMEN)

ids_for_test = 1000
output_folder = 'fasion-dataset'
output_folder_test = os.path.join(output_folder, 'test')
output_folder_train = os.path.join(output_folder, 'train')

def get_id(image_path):
    path_names = image_path.split(r'/')
    path_names[3] = path_names[3].replace('_', '')
    path_names[4] = path_names[4].split('_')[0]
    return ''.join(path_names)

def get_pose_name(image_path):
    path_names = image_path.split(r'/')
    return ''.join(path_names[4].split('_')[1:])

images = []
for root, _, files in os.walk('fasion'):
    images += [os.path.join(root, name) for name in files]
ids = {get_id(image_path) for image_path in images}

np.random.seed(0)
test_ids = set(np.random.choice(list(ids), size=ids_for_test, replace=False))

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_folder_train):
    os.makedirs(output_folder_train)

if not os.path.exists(output_folder_test):
    os.makedirs(output_folder_test)

for image_path in images:
    id = get_id(image_path)
    pose = get_pose_name(image_path)
    result_name = id + '_' + pose
    output_folder_for_image = output_folder_test if id in test_ids else output_folder_train
    shutil.copy(image_path, dst=os.path.join(output_folder_for_image, result_name))
