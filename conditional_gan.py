from keras.models import Model, Input
from keras.layers import Flatten, Concatenate, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from tensorflow.contrib.image import transform as tf_affine_transform

from gan.gan import GAN

from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras.backend import tf as ktf
import numpy as np

class WarpLayer(Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(WarpLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_size = list(input_shape[0][1:3])
        self.sc_shape = list(input_shape[0])
        self.sc_shape[0] = self.batch_size

    def call(self, inputs):
        index = inputs[1]
        index = ktf.image.resize_images(index, self.image_size)
        index = index * np.array(self.image_size, dtype='float32').reshape((1, 1, 1, 2))
        index = ktf.cast(index, 'int32')

        batch_index = ktf.range(self.batch_size)
        batch_index = ktf.reshape(batch_index, [-1] + [1] * len(self.image_size) + [1])
        batch_index = ktf.tile(batch_index, [1] + self.image_size + [1])

        index = ktf.concat([batch_index, index], axis=-1)
        return ktf.scatter_nd(index, inputs[0], self.sc_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {"batch_size": self.batch_size}
        base_config = super(WarpLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AffineTransformLayer(Layer):
    def __init__(self, affine_number, **kwargs):
        self.affine_number = affine_number
        super(AffineTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_size = list(input_shape[0][1:])

    def call(self, inputs):
        expanded_tensor = ktf.expand_dims(inputs[0], -1)
        multiples = [1, self.affine_number, 1, 1, 1]
        tiled_tensor = ktf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = ktf.reshape(tiled_tensor, ktf.shape(inputs[0]) * np.array([self.affine_number, 1, 1, 1]))
        affine_transforms = ktf.reshape(inputs[1], (-1, 8))
        tranformed = tf_affine_transform(repeated_tensor, affine_transforms)
        res = ktf.reshape(tranformed, [-1, self.affine_number] + self.image_size)
        res = ktf.transpose(res, [0, 2, 3, 1, 4])
        return ktf.reshape(res, [-1] + self.image_size[:2] + [self.image_size[2] * self.affine_number])

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0][0]] + self.image_size[:2] + [self.image_size[2] * self.affine_number])

    def get_config(self):
        config = {"affine_number": self.affine_number}
        base_config = super(AffineTransformLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
    
    
def make_generator(image_size, use_input_pose, warp_skip, batch_size):
    # input is 128 x 64 x nc
    use_warp_skip = warp_skip != 'none'
    input_img = Input(list(image_size) + [3])
    output_pose = Input(list(image_size) + [18])
    input_pose = Input(list(image_size) + [18])
    output_img = Input(list(image_size) + [3])
    if warp_skip == 'mul':
        warp = Input((10, 8))
    else:
        warp = Input(list(image_size) + [2])
    encoder_dim = 64 if warp_skip != 'mul' else 8

    inputs = [input_img]
    if use_input_pose:
        inputs.append(input_pose)
    if not use_warp_skip:
        inputs.append(output_pose)
    inp = Concatenate(axis=-1)(inputs)

    e1 = ZeroPadding2D((1, 1)) (inp)
    e1 = Conv2D(encoder_dim, kernel_size=(4, 4), strides=(2, 2))(e1)
    #input is 64 x 32 x 64
    e2 = block(e1, encoder_dim * 2)
    #input is 32 x 16 x 128
    e3 = block(e2, encoder_dim * 4)
    #input is 16 x 8 x 256
    e4 = block(e3, encoder_dim * 8)
    #input is 8 x 4 x 512
    e5 = block(e4, encoder_dim * 8)
    #input is 4 x 2 x 512
    e6 = block(e5, encoder_dim * 8, bn=False)
    #input is 2 x 1 x 512

    if use_warp_skip:
        e1_pose = ZeroPadding2D((1, 1)) (output_pose)
        e1_pose = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(e1_pose)
        #input is 64 x 32 x 64
        e2_pose = block(e1_pose, 128)
        #input is 32 x 16 x 128
        e3_pose = block(e2_pose, 256)
        #input is 16 x 8 x 256
        e4_pose = block(e3_pose, 512)
        #input is 8 x 4 x 512
        e5_pose = block(e4_pose, 512)
        #input is 4 x 2 x 512
        e6_pose = block(e5_pose, 512, bn=False)

        if warp_skip == 'sg':
            e1_warp = WarpLayer(batch_size)([e1, warp])
            e2_warp = WarpLayer(batch_size)([e2, warp])
            e3_warp = WarpLayer(batch_size)([e3, warp])
            e4_warp = WarpLayer(batch_size)([e4, warp])
            e5_warp = e5
            e6_warp = e6
        else:
            e1_warp = AffineTransformLayer(10)([e1, warp])
            e2_warp = AffineTransformLayer(10)([e2, warp])
            e3_warp = AffineTransformLayer(10)([e3, warp])
            e4_warp = AffineTransformLayer(10)([e4, warp])
            e5_warp = AffineTransformLayer(10)([e5, warp])
            e6_warp = AffineTransformLayer(10)([e6, warp])

        e1 = Concatenate(axis=-1)([e1_pose, e1_warp])
        e2 = Concatenate(axis=-1)([e2_pose, e2_warp])
        e3 = Concatenate(axis=-1)([e3_pose, e3_warp])
        e4 = Concatenate(axis=-1)([e4_pose, e4_warp])
        e5 = Concatenate(axis=-1)([e5_pose, e5_warp])
        e6 = Concatenate(axis=-1)([e6_pose, e6_warp])

    out = block(e6, 512, down=False, leaky=False, dropout = True)
    #input is 4 x 2 x 512  
    out = Concatenate(axis=-1)([out, e5])
    out = block(out, 512, down=False, leaky=False, dropout = True)
    #input is 8 x 4 x 512
    out = Concatenate(axis=-1)([out, e4])
    out = block(out, 512, down=False, leaky=False, dropout = True)
    #input is 16 x 8 x 512
    out = Concatenate(axis=-1)([out, e3])
    out = block(out, 512, down=False, leaky=False)
    #input is 32 x 16 x 512
    out = Concatenate(axis=-1)([out, e2])
    out = block(out, 256, down=False, leaky=False)
    #input is 64 x 32 x 256
    out = Concatenate(axis=-1)([out, e1])
    out = block(out, 3, down=False, leaky=False, bn=False)
    #input is  128 x 64 x 128
    
    out = Activation('tanh')(out)

    inputs = [input_img]
    inputs.append(output_pose)
    if use_input_pose:
        inputs.append(input_pose)

    return Model(inputs=inputs + ([output_img, warp] if use_warp_skip else [output_img]), outputs=inputs + [out])


def make_discriminator(image_size, use_input_pose):
    input_img = Input(list(image_size) + [3])
    output_pose = Input(list(image_size) + [18])
    input_pose = Input(list(image_size) + [18])
    output_img = Input(list(image_size) + [3])

    inputs = [input_img, output_pose]
    if use_input_pose:
        inputs.append(input_pose)
    inputs.append(output_img)

    out = Concatenate(axis=-1)(inputs)
    out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
    out = block(out, 128)
    out = block(out, 256)
    out = block(out, 512)
    out = block(out, 1, bn=False)
    out = Activation('sigmoid')(out)
    out = Flatten()(out)
    return Model(inputs=inputs, outputs=[out])


class CGAN(GAN):
    def __init__(self, generator, discriminator, l1_penalty_weight, use_input_pose, **kwargs):
        super(CGAN, self).__init__(generator, discriminator, generator_optimizer=Adam(2e-4, 0.5, 0.999),
                                    discriminator_optimizer=Adam(2e-4, 0.5, 0.999), **kwargs)
     
        self._l1_penalty_weight= l1_penalty_weight
        self.generator_metric_names = ['gan_loss', 'l1_loss']
        self._use_input_pose = use_input_pose

    def _compile_generator_loss(self):
        gan_loss_fn = super(CGAN, self)._compile_generator_loss()[0]
        if self._use_input_pose:
            l1_loss = self._l1_penalty_weight* K.mean(K.abs(self._generator_input[3] - self._discriminator_fake_input[3]))
        else:
            l1_loss = self._l1_penalty_weight * K.mean(K.abs(self._generator_input[2] - self._discriminator_fake_input[2]))

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
