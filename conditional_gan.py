from keras.models import Model
from keras.layers import Flatten, Concatenate, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K

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
        return input_shape[1]

    def get_config(self):
        config = {"batch_size": self.batch_size}
        base_config = super(WarpLayer, self).get_config()
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
    
    
def make_generator(input_a, input_b):
    # input is 128 x 64 x nc
    e1 = ZeroPadding2D((1, 1)) (input_a)
    e1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(e1)
    #input is 64 x 32 x 64
    e2 = block(e1, 128)
    #input is 32 x 16 x 128
    e3 = block(e2, 256)
    #input is 16 x 8 x 256
    e4 = block(e3, 512)
    #input is 8 x 4 x 512
    e5 = block(e4, 512)
    #input is 4 x 2 x 512
    e6 = block(e5, 512, bn = False)
    #input is 2 x 1 x 512
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
    
    out = Activation('tanh') (out)
    
    return Model(inputs=[input_a, input_b], outputs=[input_a, out])


def make_discriminator(input_a, input_b):
    out = Concatenate(axis=-1)([input_a, input_b])
    out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
    out = block(out, 128)
    out = block(out, 256)
    out = block(out, 512)
    out = block(out, 1, bn=False)
    out = Activation('sigmoid')(out)
    out = Flatten()(out)
    return Model(inputs=[input_a, input_b], outputs=[out])


class CGAN(GAN):
    def __init__(self, generator, discriminator, pose_estimator, l1_weigh_penalty=100, **kwargs):
        super(CGAN, self).__init__(generator, discriminator, generator_optimizer=Adam(2e-4, 0.5, 0.999),
                                    discriminator_optimizer=Adam(2e-4, 0.5, 0.999), **kwargs)
     
        self._l1_weigh_penalty = l1_weigh_penalty
        self.generator_metric_names = ['gan_loss', 'l1_loss', 'pose_loss']
        self._pose_penalty_weight = pose_penalty_weight
        self._pose_estimator = pose_estimator

    def _compile_generator_loss(self):
        gan_loss_fn = super(CGAN, self)._compile_generator_loss()[0]
        l1_loss = self._l1_weigh_penalty * K.mean(K.abs(self._generator_input[1] - self._discriminator_fake_input[1]))

        def l1_loss_fn(y_true, y_pred):
            return l1_loss
        
        def generator_loss(y_true, y_pred):
            return gan_loss_fn(y_true, y_pred) + l1_loss_fn(y_true, y_pred)
        return generator_loss, [gan_loss_fn, l1_loss_fn]


