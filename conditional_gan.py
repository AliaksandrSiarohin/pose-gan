from keras.models import Model, Input, Sequential
from keras.layers import Flatten, Concatenate, Activation, Dropout, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.backend import tf as ktf

from gan.gan import GAN
from gan.layer_utils import content_features_model

from keras.optimizers import Adam
from pose_transform import AffineTransformLayer


def block(out, nkernels, down=True, bn=True, dropout=False, leaky=True):
    if leaky:
        out = LeakyReLU(0.2)(out)
    else:
        out = Activation('relu')(out)
    if down:
        out = ZeroPadding2D((1, 1))(out)
        out = Conv2D(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
    else:
        out = Conv2DTranspose(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
        out = Cropping2D((1, 1))(out)
    if bn:
        out = InstanceNormalization()(out)
    if dropout:
        out = Dropout(0.5)(out)
    return out


def encoder(inps, nfilters=(64, 128, 256, 512, 512, 512)):
    layers = []
    if len(inps) != 1:
        out = Concatenate(axis=-1)(inps)
    else:
        out = inps[0]
    for i, nf in enumerate(nfilters):
        if i == 0:
            out = Conv2D(nf, kernel_size=(3, 3), padding='same')(out)
        elif i == len(nfilters) - 1:
            out = block(out, nf, bn=False)
        else:
            out = block(out, nf)
        layers.append(out)
    return layers


def decoder(skips, nfilters=(512, 512, 512, 256, 128, 3)):
    out = None
    for i, (skip, nf) in enumerate(zip(skips, nfilters)):
        if 0 < i < 3:
            out = Concatenate(axis=-1)([out, skip])
            out = block(out, nf, down=False, leaky=False, dropout=True)
        elif i == 0:
            out = block(skip, nf, down=False, leaky=False, dropout=True)
        elif i == len(nfilters) - 1:
            out = Concatenate(axis=-1)([out, skip])
            out = Activation('relu')(out)
            out = Conv2D(nf, kernel_size=(3, 3), use_bias=True, padding='same')(out)
        else:
            out = Concatenate(axis=-1)([out, skip])
            out = block(out, nf, down=False, leaky=False)
    out = Activation('tanh')(out)
    return out


from stn import SpatialTransformer
import numpy as np


def concatenate_skips(skips_app, skips_pose, warp, image_size, warp_agg, warp_skip):
    skips = []
    if warp_skip == 'stn':
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((32, 6), dtype='float32')
        weights = [W, b.flatten()]

        locnet = Sequential()
        locnet.add(Dense(64, input_shape=(72, )))
        locnet.add(LeakyReLU(0.2))
        locnet.add(Dense(32))
        locnet.add(LeakyReLU(0.2))
        locnet.add(Dense(6, weights=weights))

    for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
        if i < 4:
            if warp_skip != 'stn':
                out = AffineTransformLayer(10 if warp_skip == 'mask' else 1, warp_agg, image_size)([sk_app] + warp)
            else:
                out = SpatialTransformer(locnet, K.int_shape(sk_app)[1:3])(warp + [sk_app])
            out = Concatenate(axis=-1)([out, sk_pose])
        else:
            out = Concatenate(axis=-1)([sk_app, sk_pose])
        skips.append(out)
    return skips

def make_generator(image_size, use_input_pose, warp_skip, disc_type, warp_agg, use_bg, pose_rep_type):
    use_warp_skip = warp_skip != 'none'
    input_img = Input(list(image_size) + [3])
    output_pose = Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])
    output_img = Input(list(image_size) + [3])
    bg_img = Input(list(image_size) + [3]) 
   
    nfilters_decoder = (512, 512, 512, 256, 128, 3) if max(image_size) == 128 else (512, 512, 512, 512, 256, 128, 3)
    nfilters_encoder = (64, 128, 256, 512, 512, 512) if max(image_size) == 128 else (64, 128, 256, 512, 512, 512, 512)

    if warp_skip == 'full':
        warp = [Input((1, 8))]
    elif warp_skip == 'mask':
        warp = [Input((10, 8)), Input((10, image_size[0], image_size[1]))]
    elif warp_skip == 'stn':
        warp = [Input((72,))]
    else:
        warp = []

    if use_input_pose:
        input_pose = [Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])]
    else:
        input_pose = []

    if use_bg:
       bg_img = [bg_img]
    else:
       bg_img = [] 

    if use_warp_skip:
        enc_app_layers = encoder([input_img] + input_pose, nfilters_encoder)
        enc_tg_layers = encoder([output_pose] + bg_img, nfilters_encoder)
        enc_layers = concatenate_skips(enc_app_layers, enc_tg_layers, warp, image_size, warp_agg, warp_skip)
    else:
        enc_layers = encoder([input_img] + input_pose + [output_pose], nfilters_encoder)

    out = decoder(enc_layers[::-1], nfilters_decoder)

    warp_in_disc = [] if disc_type != 'warp' else warp

    return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + bg_img + warp,
                 outputs=[input_img] + input_pose + [out, output_pose] + bg_img + warp_in_disc)


def make_discriminator(image_size, use_input_pose, warp_skip, disc_type, warp_agg, use_bg, pose_rep_type):
    input_img = Input(list(image_size) + [3])
    output_pose = Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])
    input_pose = Input(list(image_size) + [18 if pose_rep_type == 'hm' else 3])
    output_img = Input(list(image_size) + [3])
    bg_img = Input(list(image_size) + [3]) 


    if warp_skip == 'full':
        warp = [Input((10, 8))]
    elif warp_skip == 'mask':
        warp = [Input((10, 8)), Input((10, image_size[0], image_size[1]))]
    else:
        warp = []

    if use_input_pose:
        input_pose = [input_pose]
    else:
        input_pose = []

    if use_bg:
       bg_img = [bg_img]
    else:
       bg_img = []

    assert (not use_bg) or (disc_type == 'call')

    if disc_type == 'call':
        out = Concatenate(axis=-1)([input_img] + input_pose + [output_img, output_pose] + bg_img)
        out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
        out = block(out, 128)
        out = block(out, 256)
        out = block(out, 512)
        out = block(out, 1, bn=False)
        out = Activation('sigmoid')(out)
        out = Flatten()(out)
        return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + bg_img, outputs=[out])
    elif disc_type == 'sim':
        out = Concatenate(axis=-1)([output_img, output_pose])
        out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
        out = block(out, 128)
        out = block(out, 256)
        out = block(out, 512)
        m_share = Model(inputs=[output_img, output_pose], outputs=[out])
        output_feat = m_share([output_img, output_pose])
        input_feat = m_share([input_img] + input_pose)

        out = Concatenate(axis=-1)([output_feat, input_feat])
        out = LeakyReLU(0.2)(out)
        out = Flatten()(out)
        out = Dense(1)(out)
        out = Activation('sigmoid')(out)

        return Model(inputs=[input_img] + input_pose + [output_img, output_pose], outputs=[out])
    else:
        out_inp = Concatenate(axis=-1)([input_img] + input_pose)
        out_inp = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out_inp)

        out_inp = AffineTransformLayer(10, warp_agg, image_size)([out_inp] + warp)

        out = Concatenate(axis=-1)([output_img, output_pose])
        out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)

        out = Concatenate(axis=-1)([out, out_inp])

        out = block(out, 128)
        out = block(out, 256)
        out = block(out, 512)
        out = block(out, 1, bn=False)
        out = Activation('sigmoid')(out)
        out = Flatten()(out)
        return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + warp, outputs=[out])


def total_variation_loss(x, image_size):
    img_nrows, img_ncols = image_size
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def nn_loss(reference, target, neighborhood_size=(3, 3)):
    v_pad = neighborhood_size[0] / 2
    h_pad = neighborhood_size[1] / 2
    val_pad = ktf.pad(reference, [[0, 0], [v_pad, v_pad], [h_pad, h_pad], [0, 0]],
                      mode='CONSTANT', constant_values=-10000)
    reference_tensors = []
    for i_begin in range(0, neighborhood_size[0]):
        i_end = i_begin - neighborhood_size[0] + 1
        i_end = None if i_end == 0 else i_end
        for j_begin in range(0, neighborhood_size[1]):
            j_end = j_begin - neighborhood_size[0] + 1
            j_end = None if j_end == 0 else j_end
            sub_tensor = val_pad[:, i_begin:i_end, j_begin:j_end, :]
            reference_tensors.append(ktf.expand_dims(sub_tensor, -1))
    reference = ktf.concat(reference_tensors, axis=-1)
    target = ktf.expand_dims(target, axis=-1)

    abs = ktf.abs(reference - target)
    norms = ktf.reduce_sum(abs, reduction_indices=[-2])
    loss = ktf.reduce_min(norms, reduction_indices=[-1])

    return loss


class CGAN(GAN):
    def __init__(self, generator, discriminator, l1_penalty_weight, gan_penalty_weight, use_input_pose, image_size,
                 content_loss_layer, tv_penalty_weight, nn_loss_area_size, lstruct_penalty_weight, **kwargs):
        super(CGAN, self).__init__(generator, discriminator, generator_optimizer=Adam(2e-4, 0.5, 0.999),
                                   discriminator_optimizer=Adam(2e-4, 0.5, 0.999), **kwargs)
        generator.summary()
        self._l1_penalty_weight = l1_penalty_weight
        self.generator_metric_names = ['gan_loss', 'l1_loss', 'tv_loss', 'lstruct']
        self._use_input_pose = use_input_pose
        self._image_size = image_size
        self._content_loss_layer = content_loss_layer
        self._gan_penalty_weight = gan_penalty_weight
        self._tv_penalty_weight = tv_penalty_weight
        self._nn_loss_area_size = nn_loss_area_size
        if lstruct_penalty_weight != 0:
            from keras.models import load_model
            self._pose_estimator = load_model(kwargs['pose_estimator'])
        self._lstruct_penalty_weight = lstruct_penalty_weight

    def _compile_generator_loss(self):
        image_index = 2 if self._use_input_pose else 1

        def st_loss(a, b):
            if self._nn_loss_area_size > 1:
                return nn_loss(a, b, (self._nn_loss_area_size, self._nn_loss_area_size))
            else:
                return K.mean(K.abs(a - b))

        if self._content_loss_layer != 'none':
            layer_name = self._content_loss_layer.split(',')
            cf_model = content_features_model(self._image_size, layer_name)
            reference = cf_model(self._generator_input[image_index])
            target = cf_model(self._discriminator_fake_input[image_index])
            l1_loss = K.constant(0)
            if type(reference) != list:
                reference = [reference]
                target = [target]
            for a, b in zip(reference, target):
                l1_loss = l1_loss + self._l1_penalty_weight * st_loss(a, b)
        else:
            reference = self._generator_input[image_index]
            target = self._discriminator_fake_input[image_index]
            l1_loss = self._l1_penalty_weight * st_loss(reference, target)

        if self._lstruct_penalty_weight != 0:
            target_struct = self._pose_estimator(self._generator_input[image_index][..., ::-1] / 2)[1][..., :18]
            struct = self._pose_estimator(self._discriminator_fake_input[image_index][..., ::-1] / 2)[1][..., :18]
            struct_loss = self._lstruct_penalty_weight * K.mean((target_struct - struct) ** 2)
        else:
            struct_loss = K.constant(0)

        def struct_loss_fn(y_true, y_pred):
            return struct_loss

        def tv_loss(y_true, y_pred):
            return self._tv_penalty_weight * total_variation_loss(self._discriminator_fake_input[image_index],
                                                                  self._image_size)

        def l1_loss_fn(y_true, y_pred):
            return l1_loss

        def gan_loss_fn(y_true, y_pred):
            loss = super(CGAN, self)._compile_generator_loss()[0](y_true, y_pred)
            return K.constant(0) if self._gan_penalty_weight == 0 else self._gan_penalty_weight * loss

        def generator_loss(y_true, y_pred):
            return gan_loss_fn(y_true, y_pred) + l1_loss_fn(y_true, y_pred) + tv_loss(y_true, y_pred) + struct_loss_fn(
                y_true, y_pred)

        return generator_loss, [gan_loss_fn, l1_loss_fn, tv_loss, struct_loss_fn]

        # def compile_models(self):
        #     if self._use_input_pose:
        #         self._discriminator_fake_input = self._generator(self._generator_input)[:4]
        #     else:
        #         self._discriminator_fake_input = self._generator(self._generator_input)[:3]
        #     if type(self._discriminator_fake_input) != list:
        #         self._discriminator_fake_input = [self._discriminator_fake_input]
        #     return self._compile_generator(), self._compile_discriminator()
