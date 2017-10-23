import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import circle, line_aa
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import skimage.measure, skimage.transform


LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1


class CordinatesWarp(object):
    @staticmethod
    def give_name_to_keypoints(array):
        res = defaultdict(lambda x:None)
        for i, name in enumerate(LABELS):
            if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
                res[name] = array[i][::-1]
        return res

    @staticmethod
    def check_valid(kp):
        return CordinatesWarp.check_keypoints_present(kp, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])

    @staticmethod
    def compute_st_distance(kp):
        st_distance1 = np.sum((kp['Rhip'] - kp['Rsho']) ** 2)
        st_distance2 = np.sum((kp['Lhip'] - kp['Lsho']) ** 2)
        return np.sqrt((st_distance1 + st_distance2)/2.0)

    @staticmethod
    def check_keypoints_present(kp, kp_names):
        result = True
        for name in kp_names:
            result = result and (name in kp)
        return result

    @staticmethod
    def body_poly(kp, st):
        return np.array([[kp['Rhip'][0] - 0.25*st, kp['Rhip'][1]],
                          kp['Rsho'],
                          kp['Lsho'],
                         [kp['Lhip'][0] + 0.25*st, kp['Lhip'][1]]])

    @staticmethod
    def joint_poly(kp, st, fr, to, sz_fr, sz_to):
        return np.array([[kp[fr][0] - sz_fr[0] * st/2.0, kp[fr][1] - sz_fr[1] * st/2.0],
                         [kp[to][0] - sz_to[0] * st/2.0, kp[to][1] + sz_to[1] * st/2.0],
                         [kp[to][0] + sz_to[0] * st/2.0, kp[to][1] + sz_to[1] * st/2.0],
                         [kp[fr][0] + sz_fr[0] * st/2.0, kp[fr][1] - sz_fr[1] * st/2.0 ]])

    @staticmethod
    def head_poly(kp, st):
        if 'neck' in kp:
            bot = kp['neck'][1]
        else:
            bot = kp['nose'][1] + 0.25 * st
        top = kp['nose'][1] - 0.25 * st

        right = kp['nose'][0] - 0.25 * st
        left = kp['nose'][0] + 0.25 * st

        return np.array([[right, bot],
                         [right, top],
                         [left, top],
                         [left, bot]])


    @staticmethod
    def apply_transform(poly1, poly2, warp, bg_mask):
        c1 = skimage.measure.grid_points_in_poly(warp.shape[0:2], poly1[..., ::-1])
        c2 = skimage.measure.grid_points_in_poly(warp.shape[0:2], poly2[..., ::-1])
        bg_mask[:] = np.logical_or(bg_mask, c1)
        bg_mask[:] = np.logical_or(bg_mask, c2)

        tr = skimage.transform.estimate_transform('affine', src=poly1[..., ::-1], dst=poly2[..., ::-1])

        yy, xx = np.meshgrid(range(warp.shape[0]), range(warp.shape[1]), indexing='ij')
        yy = yy.reshape((-1, ))
        xx = xx.reshape((-1, ))

        new_cords = tr(np.vstack([yy, xx]).T)
        new_cords = new_cords.reshape((warp.shape[0], warp.shape[1], 2))

        warp[c1] = new_cords[c1]

    @staticmethod
    def apply_pose_transforms(kp1, kp2, st1, st2, img_size):
        warp = -np.ones(list(img_size) + [2])
        bg_mask = np.zeros(list(img_size), dtype='bool')

        body_poly_1 = CordinatesWarp.body_poly(kp1, st1)
        body_poly_2 = CordinatesWarp.body_poly(kp2, st2)
        CordinatesWarp.apply_transform(body_poly_1, body_poly_2, warp, bg_mask)

        #estimate head
        head_present_1 = 'nose' in kp1
        head_present_2 = 'nose' in kp2
        if head_present_1 and head_present_2:
            head_poly_1 = CordinatesWarp.head_poly(kp1, st1)
            head_poly_2 = CordinatesWarp.head_poly(kp2, st2)
            CordinatesWarp.apply_transform(head_poly_1, head_poly_2, warp, bg_mask)


        def joint_transform(fr, to, sz_fr, sz_to):
            if not CordinatesWarp.check_keypoints_present(kp2, [fr, to]):
                return
            if not CordinatesWarp.check_keypoints_present(kp1, [fr, to]):
                return
            poly_2 = CordinatesWarp.joint_poly(kp2, st2, fr, to, sz_fr, sz_to)
            poly_1 = CordinatesWarp.joint_poly(kp1, st1, fr, to, sz_fr, sz_to)

            print (poly_1)
            CordinatesWarp.apply_transform(poly_1, poly_2, warp, bg_mask)

        joint_transform('Rhip', 'Rkne', [0.5, 0], [0.25, 0])
        joint_transform('Lhip', 'Lkne', [0.5, 0], [0.25, 0])

        joint_transform('Rkne', 'Rank', [0.25, 0], [0.5, 0.5])
        joint_transform('Lkne', 'Lank', [0.25, 0], [0.5, 0.5])

        joint_transform('Rsho', 'Relb', [0.25, 0], [0.25, 0])
        joint_transform('Lsho', 'Lelb', [0.25, 0], [0.25, 0])

        joint_transform('Relb', 'Rwri', [0.25, 0], [0.25, 0.0])
        joint_transform('Lelb', 'Lwri', [0.25, 0], [0.25, 0.0])

        return warp, np.logical_not(bg_mask)

    @staticmethod
    def background_tranforms(warp, bg_mask):
        yy, xx = np.meshgrid(range(warp.shape[0]), range(warp.shape[1]), indexing='ij')
        i_tr = np.concatenate([yy[..., np.newaxis], xx[..., np.newaxis]], axis=-1)
        i_tr = i_tr.astype(dtype='float32')
        warp[bg_mask] = i_tr[bg_mask]

    @staticmethod
    def warp_mask(array1, array2, img_size):
        kp1 = CordinatesWarp.give_name_to_keypoints(array1)
        kp2 = CordinatesWarp.give_name_to_keypoints(array2)

        st1 = CordinatesWarp.compute_st_distance(kp1)
        st2 = CordinatesWarp.compute_st_distance(kp2)

        warp, bg_mask = CordinatesWarp.apply_pose_transforms(kp1, kp2, st1, st2, img_size)
        CordinatesWarp.background_tranforms(warp, bg_mask)

        warp[..., 0] /= warp.shape[0]
        warp[..., 1] /= warp.shape[1]

        return warp, bg_mask


def map_to_cord(pose_map, threshold = 0.1):
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[..., :18]
       
    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)


def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result


def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def draw_pose_from_map(pose_map, threshold=0.1, **kwargs):
    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def mean_inputation(X):
    X = X.copy()
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            val = np.mean(X[:, i, j][X[:, i, j] != -1])
            X[:, i, j][X[:, i, j] == -1] = val
    return X

def draw_legend():
    handles = [mpatches.Patch(color=np.array(color) / 255.0, label=name) for color, name in zip(COLORS, LABELS)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

if __name__ == "__main__":
    import pandas as pd
    from skimage.io import imread
    import pylab as plt
    import os
    i = 5
    df = pd.read_csv('cao-hpe/annotations.csv', sep=':')
    output_folder = 'train-annotated'
    input_folder = '../market-dataset/bounding_box_train'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for index, row in df.iterrows():
        if row['name'] != '0002_c1s1_000551_01.jpg':
            continue
        img = imread(os.path.join(input_folder, row['name']))
        print (row['name'])
        pose_cords = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])

        map = cords_to_map(pose_cords, (128, 64))

        colors, mask = draw_pose_from_map(map)

#         img[mask] = colors[mask]

#         a_not_resized=np.load(row['name'] + '.npy')
#         #a_resized = resize(a_not_resized, output_shape=(128, 64), preserve_range=True)
#         print (pose_cords[1])
#         print (map[..., 1].mean())

#         from scipy.optimize import minimize_scalar
#         fn = lambda sigma: np.sum((cords_to_map(pose_cords, (128, 64), sigma) - a_not_resized[..., :18]) ** 2)
#         a = minimize_scalar(fn, bounds=(0, 21))
#         print (a)

        #print (map.max(axis=2))
        plt.imshow(colors, cmap=plt.cm.gray_r)
        draw_legend()
        plt.savefig(os.path.join(output_folder, row['name'] +'_joints.png'))

        # plt.imshow(a_not_resized[..., 1], cmap=plt.cm.gray_r)
        # draw_legend()
        # plt.savefig(os.path.join(output_folder, row['name'] +'_img_joints.png'))
