from keras.models import Model, Input
from keras.layers import Dense, Reshape, Flatten, Concatenate, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K

from gan.gan import GAN
from gan.dataset import UGANDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer

import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
import os

from keras.optimizers import Adam


def block(out, nkernels, down = True, bn = True, dropout = False, leaky=True):
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
    e6 = block(e5, 512)
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
    out = block(out, 3, down=False, leaky=False)
    #input is  128 x 64 x 128
    
    out = Activation('tanh') (out)
    
    return Model(inputs=[input_a, input_b], outputs=[input_a, out])


def make_discriminator(input_a, input_b):
    out = Concatenate(axis=-1)([input_a, input_b])
    out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
    out = block(out, 128)
    out = block(out, 256)
    out = block(out, 1, bn=False)
    #out = block(out, 1, bn=False)
    out = Activation('sigmoid')(out)
    out = Flatten()(out)
    return Model(inputs=[input_a, input_b], outputs=[out])


class CGAN(GAN):
    def __init__(self, generator, discriminator, l1_weigh_penalty = 100, **kwargs):
        super(CGAN, self).__init__(generator, discriminator, generator_optimizer = Adam(2e-4, 0.5, 0.999),
                                                 discriminator_optimizer = Adam(2e-4, 0.5, 0.999), **kwargs)
        self._l1_weigh_penalty = l1_weigh_penalty
        self.generator_metric_names = ['gan_loss', 'l1_loss']

    def _compile_generator_loss(self):
        gan_loss_fn = super(CGAN, self)._compile_generator_loss()[0]
        l1_loss = self._l1_weigh_penalty * K.mean(K.abs(self._generator_input[1] - self._discriminator_fake_input[1]))
        def l1_loss_fn(y_true, y_pred):
            return l1_loss
        
        def generator_loss(y_true, y_pred):
            return gan_loss_fn(y_true, y_pred) + l1_loss_fn(y_true, y_pred)
        return generator_loss, [gan_loss_fn, l1_loss_fn]

class ConditionalDataset(UGANDataset):
    def __init__(self, batch_size, input_folder, pose_estimator, img_size_final, img_size_init):
        super(ConditionalDataset, self).__init__(batch_size, None)
        self._input_folder = input_folder
        self._pose_estimator = pose_estimator
        self._folder_b = os.path.join(input_folder, 'A')
        self._names = np.array(os.listdir(self._folder_a))
        self._img_size_final = img_size_final
        self._img_size_init = img_size_init
        self._batches_before_shuffle = len(self._names) / self._batch_size
        
    def _random_crop(self, img_1, img_2):
        size=self._img_size_final
        y = np.random.randint(img_1.shape[0] - size[0], size=1)[0]
        x = np.random.randint(img_1.shape[1] - size[1], size=1)[0]
        return img_1[y:(y+size[0]), x:(x+size[1])], img_2[y:(y+size[0]), x:(x+size[1])]
        
    def _load_data_batch(self, index):
        load_from_folder = lambda folder: [resize(imread(os.path.join(folder, name)), self._img_size_init, preserve_range = True) 
                                          for name in self._names[index]]
        a = load_from_folder(self._folder_a)
        b = load_from_folder(self._folder_b)
        
        ab_resized = [self._random_crop(img_a, img_b) for img_a, img_b in zip(a,b)]
        a_batch, b_batch = zip(*ab_resized)
        
        result = [self._preprocess(np.array(a_batch)), self._preprocess(np.array(b_batch))]
        return result
    
    def _preprocess(self, image):
        return (image / 255 - 0.5) * 2
    
    def _deprocess(self, image):
        return ((image/2 + 0.5) * 255).astype(np.uint8)
        
    def next_generator_sample(self):
        index = np.random.choice(len(self._names), replace=False, size = (self._batch_size,))
        return self._load_data_batch(index)
    
    def _load_discriminator_data(self, index):
        return self._load_data_batch(index)     
        
    
    def _shuffle_discriminator_data(self):
        np.random.shuffle(self._names) 
        
        
    def display(self, output_batch, input_batch = None, row=1, col=1):
        a_gen_images = output_batch[1]        
        a_real_images = output_batch[0]
        b_images = input_batch[1]
        
        a_gen_res = super(ConditionalDataset, self).display(a_gen_images, row=row, col=col)
        b_true_res = super(ConditionalDataset, self).display(b_images, row=row, col=col)
        a_true_res = super(ConditionalDataset, self).display(a_real_images, row=row, col=col)
        result = self._deprocess(np.concatenate([a_gen_res, a_true_res, b_true_res], axis=1))
        return result


def main():
    generator = make_generator(Input((256, 256, 3)), Input((256, 256, 3)))
    discriminator = make_discriminator(Input((256, 256, 3)), Input((256, 256, 3)))

    args = parser_with_default_args().parse_args()
    dataset = ConditionalDataset(args.batch_size, 'datasets/facades_a_b', (256, 256), (286, 286))
    gan = CGAN(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()

if __name__ == "__main__":
    main()
