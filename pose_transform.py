from keras.models import Input, Model
from keras.engine.topology import Layer
from keras.backend import tf as ktf


import pose_utils
import pylab as plt
import numpy as np
from skimage.io import imread
from skimage.transform import warp_coords
import skimage.draw

import skimage.measure
import skimage.transform


from pose_utils import LABELS, MISSING_VALUE
from tensorflow.contrib.image import transform as tf_affine_transform


class AffineTransformLayer(Layer):
    def __init__(self, number_of_transforms, aggregation_fn, init_image_size, **kwargs):
        assert aggregation_fn in ['none', 'max', 'avg']
        self.aggregation_fn = aggregation_fn
        self.number_of_transforms = number_of_transforms
        self.init_image_size = init_image_size
        super(AffineTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_size = list(input_shape[0][1:])
        self.affine_mul = [1, 1, self.init_image_size[0] / self.image_size[0],
                           1, 1, self.init_image_size[1] / self.image_size[1],
                           1, 1]
        self.affine_mul = np.array(self.affine_mul).reshape((1, 1, 8))

    def call(self, inputs):
        expanded_tensor = ktf.expand_dims(inputs[0], -1)
        multiples = [1, self.number_of_transforms, 1, 1, 1]
        tiled_tensor = ktf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = ktf.reshape(tiled_tensor, ktf.shape(inputs[0]) * np.array([self.number_of_transforms, 1, 1, 1]))

        affine_transforms = inputs[1] / self.affine_mul

        affine_transforms = ktf.reshape(affine_transforms, (-1, 8))
        tranformed = tf_affine_transform(repeated_tensor, affine_transforms)
        res = ktf.reshape(tranformed, [-1, self.number_of_transforms] + self.image_size)
        res = ktf.transpose(res, [0, 2, 3, 1, 4])

        #Use masks
        if len(inputs) == 3:
            mask = ktf.transpose(inputs[2], [0, 2, 3, 1])
            mask = ktf.image.resize_images(mask, self.image_size[:2], method=ktf.image.ResizeMethod.NEAREST_NEIGHBOR)
            res = res * ktf.expand_dims(mask, axis=-1)


        if self.aggregation_fn == 'none':
            res = ktf.reshape(res, [-1] + self.image_size[:2] + [self.image_size[2] * self.number_of_transforms])
        elif self.aggregation_fn == 'max':
            res = ktf.reduce_max(res, reduction_indices=[-2])
        elif self.aggregation_fn == 'avg':
            counts = ktf.reduce_sum(mask, reduction_indices=[-1])
            counts = ktf.expand_dims(counts, axis=-1)
            res = ktf.reduce_sum(res, reduction_indices=[-2])
            res /= counts
            res = ktf.where(ktf.is_nan(res), ktf.zeros_like(res), res)
        return res

    def compute_output_shape(self, input_shape):
        if self.aggregation_fn == 'none':
            return tuple([input_shape[0][0]] + self.image_size[:2] + [self.image_size[2] * self.number_of_transforms])
        else:
            return input_shape[0]

    def get_config(self):
        config = {"number_of_transforms": self.number_of_transforms,
                  "aggregation_fn": self.aggregation_fn}
        base_config = super(AffineTransformLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def give_name_to_keypoints(array):
    res = {}
    for i, name in enumerate(LABELS):
        if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
            res[name] = array[i][::-1]
    return res


def check_valid(kp_array):
    kp = give_name_to_keypoints(kp_array)
    return check_keypoints_present(kp, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])


def check_keypoints_present(kp, kp_names):
    result = True
    for name in kp_names:
        result = result and (name in kp)
    return result


def compute_st_distance(kp):
    st_distance1 = np.sum((kp['Rhip'] - kp['Rsho']) ** 2)
    st_distance2 = np.sum((kp['Lhip'] - kp['Lsho']) ** 2)
    return np.sqrt((st_distance1 + st_distance2)/2.0)


def mask_from_kp_array(kp_array, border_inc, img_size):
    min = np.min(kp_array, axis=0)
    max = np.max(kp_array, axis=0)
    min -= int(border_inc)
    max += int(border_inc)

    min = np.maximum(min, 0)
    max = np.minimum(max, img_size[::-1])

    mask = np.zeros(img_size)
    mask[min[1]:max[1], min[0]:max[0]] = 1
    return mask


def get_array_of_points(kp, names):
    return np.array([kp[name] for name in names])


def pose_masks(array2, img_size):
    kp2 = give_name_to_keypoints(array2)
    masks = []
    st2 = compute_st_distance(kp2)
    empty_mask = np.zeros(img_size)

    body_mask = np.ones(img_size)# mask_from_kp_array(get_array_of_points(kp2, ['Rhip', 'Lhip', 'Lsho', 'Rsho']), 0.1 * st2, img_size)
    masks.append(body_mask)

    head_candidate_names = {'Leye', 'Reye', 'Lear', 'Rear', 'nose'}
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp2:
            head_kp_names.add(cn)


    if len(head_kp_names)!=0:
        center_of_mass = np.mean(get_array_of_points(kp2, list(head_kp_names)), axis=0, keepdims=True)
        center_of_mass = center_of_mass.astype(int)
        head_mask = mask_from_kp_array(center_of_mass, 0.40 * st2, img_size)
        masks.append(head_mask)
    else:
        masks.append(empty_mask)

    def mask_joint(fr, to, inc_to):
        if not check_keypoints_present(kp2, [fr, to]):
            return empty_mask
        return skimage.measure.grid_points_in_poly(img_size, estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)[:, ::-1])

    masks.append(mask_joint('Rhip', 'Rkne', 0.1))
    masks.append(mask_joint('Lhip', 'Lkne', 0.1))

    masks.append(mask_joint('Rkne', 'Rank', 0.5))
    masks.append(mask_joint('Lkne', 'Lank', 0.5))

    masks.append(mask_joint('Rsho', 'Relb', 0.1))
    masks.append(mask_joint('Lsho', 'Lelb', 0.1))

    masks.append(mask_joint('Relb', 'Rwri', 0.5))
    masks.append(mask_joint('Lelb', 'Lwri', 0.5))

    return np.array(masks)


def estimate_polygon(fr, to, st, inc_to, inc_from, p_to, p_from):
    fr = fr + (fr - to) * inc_from
    to = to + (to - fr) * inc_to

    norm_vec = fr - to
    norm_vec = np.array([-norm_vec[1], norm_vec[0]])
    norm = np.linalg.norm(norm_vec)
    if norm == 0:
        return np.array([
            fr + 1,
            fr - 1,
            to - 1,
            to + 1,
        ])
    norm_vec = norm_vec / norm
    vetexes = np.array([
        fr + st * p_from * norm_vec,
        fr - st * p_from * norm_vec,
        to - st * p_to * norm_vec,
        to + st * p_to * norm_vec
    ])

    return vetexes

def affine_transforms(array1, array2):
    kp1 = give_name_to_keypoints(array1)
    kp2 = give_name_to_keypoints(array2)

    st1 = compute_st_distance(kp1)
    st2 = compute_st_distance(kp2)


    no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])

    transforms = []
    def to_transforms(tr):
        from numpy.linalg import LinAlgError
        try:
            np.linalg.inv(tr)
            transforms.append(tr)
        except LinAlgError:
            transforms.append(no_point_tr)

    body_poly_1 = get_array_of_points(kp1, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])
    body_poly_2 = get_array_of_points(kp2, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])
    tr = skimage.transform.estimate_transform('affine', src=body_poly_2, dst=body_poly_1)

    to_transforms(tr.params)

    head_candidate_names = {'Leye', 'Reye', 'Lear', 'Rear', 'nose'}
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp1 and cn in kp2:
            head_kp_names.add(cn)
    if len(head_kp_names) != 0:
        #if len(head_kp_names) < 3:
        head_kp_names.add('Lsho')
        head_kp_names.add('Rsho')
        head_poly_1 = get_array_of_points(kp1, list(head_kp_names))
        head_poly_2 = get_array_of_points(kp2, list(head_kp_names))
        tr = skimage.transform.estimate_transform('affine', src=head_poly_2, dst=head_poly_1)
        to_transforms(tr.params)
    else:
        to_transforms(no_point_tr)

    def estimate_join(fr, to, inc_to):
        if not check_keypoints_present(kp2, [fr, to]):
            return no_point_tr
        poly_2 = estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)
        if check_keypoints_present(kp1, [fr, to]):
            poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
        else:
            if fr[0]=='R':
                fr = fr.replace('R', 'L')
                to = to.replace('R', 'L')
            else:
                fr = fr.replace('L', 'R')
                to = to.replace('L', 'R')
            if check_keypoints_present(kp1, [fr, to]):
                poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
            else:
                return no_point_tr
        return skimage.transform.estimate_transform('affine', dst=poly_1, src=poly_2).params

    to_transforms(estimate_join('Rhip', 'Rkne', 0.1))
    to_transforms(estimate_join('Lhip', 'Lkne', 0.1))

    to_transforms(estimate_join('Rkne', 'Rank', 0.3))
    to_transforms(estimate_join('Lkne', 'Lank', 0.3))

    to_transforms(estimate_join('Rsho', 'Relb', 0.1))
    to_transforms(estimate_join('Lsho', 'Lelb', 0.1))

    to_transforms(estimate_join('Relb', 'Rwri', 0.3))
    to_transforms(estimate_join('Lelb', 'Lwri', 0.3))

    return np.array(transforms).reshape((-1, 9))[..., :-1]


def estimate_uniform_transform(array1, array2):
    kp1 = give_name_to_keypoints(array1)
    kp2 = give_name_to_keypoints(array2)

    no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])

    def check_invertible(tr):
        from numpy.linalg import LinAlgError
        try:
            np.linalg.inv(tr)
            return True
        except LinAlgError:
            return False

    keypoint_names = {'Rhip', 'Lhip', 'Lsho', 'Rsho'}
    candidate_names = {'Rkne', 'Lkne'}

    for cn in candidate_names:
        if cn in kp1 and cn in kp2:
            keypoint_names.add(cn)

    poly_1 = get_array_of_points(kp1, list(keypoint_names))
    poly_2 = get_array_of_points(kp2, list(keypoint_names))

    tr = skimage.transform.estimate_transform('affine', src=poly_2, dst=poly_1)

    tr = tr.params

    if check_invertible(tr):
        return tr.reshape((-1, 9))[..., :-1]
    else:
        return no_point_tr.reshape((-1, 9))[..., :-1]


def draw_line(fr, to, thickness, shape):
    norm_vec = fr - to
    norm_vec = np.array([-norm_vec[1], norm_vec[0]])
    norm_vec = thickness * norm_vec / np.linalg.norm(norm_vec)

    vetexes = np.array([
        fr + norm_vec,
        fr - norm_vec,
        to - norm_vec,
        to + norm_vec
    ])

    return skimage.draw.polygon(vetexes[:, 1], vetexes[:, 0], shape=shape)

def make_stickman(kp_array, img_shape):
    kp = give_name_to_keypoints(kp_array)
    #Adapted from https://github.com/CompVis/vunet/
    # three channels: left, right, center
    scale_factor = img_shape[1] / 128.0
    thickness = int(3 * scale_factor)
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype="float32"))

    body = ["Lhip", "Lsho", "Rsho", "Rhip"]
    body_pts = get_array_of_points(kp, body)
    if np.min(body_pts) >= 0:
        body_pts = np.int_(body_pts)
        rr,cc = skimage.draw.polygon(body_pts[:,1], body_pts[:, 0], shape=img_shape)
        imgs[2][rr, cc] = 1

    right_lines = [
            ("Rank", "Rkne"),
            ("Rkne", "Rhip"),
            ("Rhip", "Rsho"),
            ("Rsho", "Relb"),
            ("Relb", "Rwri")]
    for line in right_lines:
        if check_keypoints_present(kp, line):
            line_pts = get_array_of_points(kp, line)
            rr,cc = draw_line(line_pts[0], line_pts[1], thickness=thickness, shape=img_shape)
            imgs[0][rr,cc] = 1

    left_lines = [
            ("Lank", "Lkne"),
            ("Lkne", "Lhip"),
            ("Lhip", "Lsho"),
            ("Lsho", "Lelb"),
            ("Lelb", "Lwri")]
    for line in left_lines:
        if check_keypoints_present(kp, line):
            line_pts = get_array_of_points(kp, line)
            rr,cc = draw_line(line_pts[0], line_pts[1], thickness=thickness, shape=img_shape)
            imgs[1][rr, cc] = 1

    if check_keypoints_present(kp, ['Rsho', 'Lsho', 'nose']):
        rs = kp["Rsho"]
        ls = kp["Lsho"]
        cn = kp["nose"]

        neck = 0.5*(rs+ls)
        a = neck
        b = cn
        if np.min(a) >= 0 and np.min(b) >= 0:
            rr,cc = draw_line(a, b, thickness=thickness, shape=img_shape)
            imgs[0][rr, cc] = 0.5
            imgs[1][rr, cc] = 0.5

    if check_keypoints_present(kp, ['Reye', 'Leye', 'nose']):
        reye = kp["Reye"]
        leye = kp["Leye"]
        cn = kp["nose"]

        neck = 0.5*(rs+ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        if np.min(a) >= 0 and np.min(b) >= 0:
            rr,cc = draw_line(cn, reye, thickness=thickness, shape=img_shape)
            imgs[0][rr, cc] = 0.5
            rr,cc = draw_line(cn, leye, thickness=thickness, shape=img_shape)
            imgs[1][rr, cc] = 0.5
    img = np.stack(imgs, axis = -1)
    return img


if __name__ == "__main__":
    import pandas as pd
    import os
    from skimage.transform import resize
    pairs_df = pd.read_csv('data/tmp-pairs-test.csv')
    kp_df = pd.read_csv('data/tmp-annotation-test.csv', sep=':')
    img_folder = 'data/tmp-dataset/test'
    f = open('lolkek.txt', 'w')
    for _, row in pairs_df.iterrows():
        print 1
        fr = 'denis_pjump000004.jpg'# row['from']
        to = 'denis_pjump000004.jpg'#row['to']
        fr_img = imread(os.path.join(img_folder, fr))
        to_img = imread(os.path.join(img_folder, to))

        kp_fr = kp_df[kp_df['name'] == fr].iloc[0]
        kp_to = kp_df[kp_df['name'] == to].iloc[0]


        plt.subplot(3, 1, 1)
        kp_fr = pose_utils.load_pose_cords_from_strings(kp_fr['keypoints_y'], kp_fr['keypoints_x'])
        kp_to = pose_utils.load_pose_cords_from_strings(kp_to['keypoints_y'], kp_to['keypoints_x'])

        img = fr_img.copy()
        #p, m = pose_utils.draw_pose_from_cords(kp_fr, img.shape[:2])
        img = make_stickman(kp_to, fr_img.shape)

        #img[m] = p[m]
        plt.imshow(img)

        plt.subplot(3, 1, 2)
        img = to_img.copy()
        p, m = pose_utils.draw_pose_from_cords(kp_to, img.shape[:2])
        img[m] = p[m]
        plt.imshow(img)

        # tr = estimate_uniform_transform(kp_fr, kp_to)
        #
        # no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])
        # if np.all(tr == no_point_tr.reshape((-1, 9))[..., :-1]):
        #     print >>f, '_'.join([fr,to]) + '.jpg'

        p = resize(p, (256, 256), preserve_range=True).astype(np.uint8)
        m = resize(m, (256, 256), preserve_range=True).astype(bool)
        fr_img = resize(fr_img, (256, 256), preserve_range=True).astype(float)

        tr = affine_transforms(kp_fr, kp_to)
        masks = pose_masks(kp_to, fr_img.shape[:2])

        x = Input(fr_img.shape)
        i = Input((10, 8))
        mm = Input([10] + list(fr_img.shape[:2]))

        y = AffineTransformLayer(10, 'max', (256, 256))([x, i, mm])
        model = Model(inputs=[x, i, mm], outputs=y)

        b = model.predict([fr_img[np.newaxis], tr[np.newaxis], masks[np.newaxis]])
        plt.subplot(3, 1, 3)

        a = b[0, ..., 0:3].copy().astype('uint8')
        a[m] = p[m]
        plt.imshow(a)

        plt.show()
        #f.flush()




    # def get(first, second):
    #     plt.subplot(3, 1, 1)
    #     a = first
    #     img = imread(a[0])
    #     image = img.copy()
    #     array1 = pose_utils.load_pose_cords_from_strings(a[2], a[1])
    #     print (img.shape)
    #     p, m = pose_utils.draw_pose_from_cords(array1, img.shape[:2])
    #     img[m] = p[m]
    #     plt.imshow(img)
    #
    #     plt.subplot(3, 1, 2)
    #     a = second
    #     img = imread(a[0])
    #     array2 = pose_utils.load_pose_cords_from_strings(a[2], a[1])
    #     p, m = pose_utils.draw_pose_from_cords(array2, img.shape[:2])
    #     img[m] = p[m]
    #     plt.imshow(img)
    #     pose_utils.draw_legend()
    #     trs = affine_transforms(array1, array2)
    #     masks = pose_masks(array2, (128, 64))
    #
    #     image = resize(image, (64, 32), preserve_range=True)
    #     m = resize(m, (64, 32), preserve_range=True, order=0).astype(bool)
    #     p = resize(p, (64, 32), preserve_range=True, order=0)
    #     return trs, masks, image, p,  m
    #
    #
    # trs, masks, img, p, m = get(first2, second2)
    # trs2, masks2, img2, p2, m2 = get(first, second)
    #
    # plt.subplot(3, 1, 3)
    #
    # x_v = np.concatenate([img[np.newaxis], img2[np.newaxis]])
    # i_v = np.concatenate([trs[np.newaxis], trs2[np.newaxis]])
    # m_v = np.concatenate([masks[np.newaxis], masks2[np.newaxis]])
    #
    # #trs = CordinatesWarp.affine_transforms(array1, array2)
    # x = Input((64,32,3))
    # i = Input((len(trs), 8))
    # masks = Input((len(trs), 128, 64))
    #
    # y = AffineTransformLayer(len(trs), 'max', (128, 64))([x, i])
    # model = Model(inputs=[x, i, masks], outputs=y)
    #
    # # x_v = skimage.transform.resize(image, (128, 64), preserve_range=True)[np.newaxis]
    # # i_v = trs[np.newaxis]
    #
    # b = model.predict([x_v, i_v, m_v])
    # print (b.shape)
    # b = b[..., :3]
    #
    #
    # # trs, _ = CordinatesWarp.warp_mask(array1, array2, img_size=img.shape[:2])
    # # x = Input((128,64,3))
    # # i = Input((128,64,2))
    # #
    # # y = WarpLayer(1)([x, i])
    # # model = Model(inputs=[x, i], outputs=y)
    # #
    # # x_v = skimage.transform.resize(image, (128, 64), preserve_range=True)[np.newaxis]
    # # i_v = trs[np.newaxis]
    # #
    # # b = model.predict([x_v, i_v])
    # # print (b.shape)
    #
    # warped_image = np.squeeze(b[1]).astype(np.uint8)
    # warped_image[m2] = p2[m2]
    # plt.imshow(warped_image)
    #
    # # from scipy.ndimage import map_coordinates
    # #
    # # mask = CordinatesWarp.warp_mask(array1, array2, (128, 64, 3))
    # # mask = np.moveaxis(mask, -1, 0)
    # # warped_image = map_coordinates(image, mask)
    # # warped_image[m] = p[m]
    # # plt.subplot(4, 1, 4)
    # # plt.imshow(warped_image)
    # plt.show()
