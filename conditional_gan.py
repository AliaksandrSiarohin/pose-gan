from keras.models import Model, Input
from keras.layers import Flatten, Concatenate, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K

from gan.gan import GAN
from gan.layer_utils import content_features_model

from keras.optimizers import Adam
from pose_transform import AffineTransformLayer


def block(out, nkernels, down=True, bn=True, dropout=False, leaky=True):
    if leaky:
        out = LeakyReLU(0.2) (out)
    else:
        out = Activation('relu') (out)
    if down:
        out = ZeroPadding2D((1, 1)) (out)
        out = Conv2D(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
    else:
        out = Conv2DTranspose(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
        out = Cropping2D((1,1))(out)
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
            out = ZeroPadding2D((1, 1))(out)
            out = Conv2D(nf, kernel_size=(4, 4), strides=(2, 2))(out)
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
            out = block(out, nf, down=False, leaky=False, bn=False)
        else:
            out = block(out, nf, down=False, leaky=False)
    out = Activation('tanh')(out)
    return out

def concatenate_skips(skips_app, skips_pose, warp, image_size):
    skips = []
    for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
        if i < 4:
            out = AffineTransformLayer(10, 'max', image_size) ([sk_app] + warp)
            out = Concatenate(axis=-1)([out, sk_pose])
        else:
            out = Concatenate(axis=-1)([sk_app, sk_pose])
        skips.append(out)
    return skips


def make_generator(image_size, use_input_pose, warp_skip):
    # input is 128 x 64 x nc
    use_warp_skip = warp_skip != 'none'
    input_img = Input(list(image_size) + [3])
    output_pose = Input(list(image_size) + [18])
    output_img = Input(list(image_size) + [3])

    nfilters_decoder = (512, 512, 512, 256, 128, 3) if max(image_size) == 128 else (512, 512, 512, 512, 256, 128, 3)
    nfilters_encoder = (64, 128, 256, 512, 512, 512) if max(image_size) == 128 else (64, 128, 256, 512, 512, 512, 512)

    if warp_skip == 'full':
        warp = [Input((10, 8))]
    elif warp_skip == 'mask':
        warp = [Input((10, 8)), Input((10, 128, 64))]
    else:
        warp = []

    if use_input_pose:
        input_pose = [Input(list(image_size) + [18])]
    else:
        input_pose = []

    if use_warp_skip:
        enc_app_layers = encoder([input_img] + input_pose, nfilters_encoder)
        enc_tg_layers = encoder([output_pose], nfilters_encoder)
        enc_layers = concatenate_skips(enc_app_layers, enc_tg_layers, warp, image_size)
    else:
        enc_layers = encoder([input_img] + input_pose + [output_pose], nfilters_encoder)

    out = decoder(enc_layers[::-1], nfilters_decoder)

    return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + warp,
                 outputs=[input_img] + input_pose + [out, output_pose])


def make_discriminator(image_size, use_input_pose):
    input_img = Input(list(image_size) + [3])
    output_pose = Input(list(image_size) + [18])
    input_pose = Input(list(image_size) + [18])
    output_img = Input(list(image_size) + [3])

    if use_input_pose:
        input_pose = [input_pose]
    else:
        input_pose = []

    out = Concatenate(axis=-1)([input_img] + input_pose + [output_img, output_pose])
    out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
    out = block(out, 128)
    out = block(out, 256)
    out = block(out, 512)
    out = block(out, 1, bn=False)
    out = Activation('sigmoid')(out)
    out = Flatten()(out)
    return Model(inputs=[input_img] + input_pose + [output_img, output_pose], outputs=[out])


class CGAN(GAN):
    def __init__(self, generator, discriminator, l1_penalty_weight, use_input_pose, image_size,
                 content_loss_layer, **kwargs):
        super(CGAN, self).__init__(generator, discriminator, generator_optimizer=Adam(2e-4, 0.5, 0.999),
                                    discriminator_optimizer=Adam(2e-4, 0.5, 0.999), **kwargs)
        generator.summary()
        self._l1_penalty_weight= l1_penalty_weight
        self.generator_metric_names = ['gan_loss', 'l1_loss']
        self._use_input_pose = use_input_pose
        self._image_size = image_size
        self._content_loss_layer = content_loss_layer

    def _compile_generator_loss(self):
        gan_loss_fn = super(CGAN, self)._compile_generator_loss()[0]

        image_index = 2 if self._use_input_pose else 1

        if self._content_loss_layer != 'none':
            cf_model = content_features_model(self._image_size, self._content_loss_layer)
            reference = cf_model(self._generator_input[image_index])
            target = cf_model(self._discriminator_fake_input[image_index])
        else:
            reference = self._generator_input[image_index]
            target = self._discriminator_fake_input[image_index]

        l1_loss = self._l1_penalty_weight * K.mean(K.abs(reference - target))

        def l1_loss_fn(y_true, y_pred):
            return l1_loss

        def generator_loss(y_true, y_pred):
            return gan_loss_fn(y_true, y_pred) + l1_loss_fn(y_true, y_pred)
        return generator_loss, [gan_loss_fn, l1_loss_fn]

    def compile_models(self):
        if self._use_input_pose:
            self._discriminator_fake_input = self._generator(self._generator_input)[:4]
        else:
            self._discriminator_fake_input = self._generator(self._generator_input)[:3]
        if type(self._discriminator_fake_input) != list:
            self._discriminator_fake_input = [self._discriminator_fake_input]
        return self._compile_generator(), self._compile_discriminator()
