from keras import backend as K
from gan.wgan_gp import WGAN_GP
from functools import partial
import numpy as np
from keras.backend import tf as ktf

class POSE_GAN(WGAN_GP):
    """
        Class for representing pose_gan
    """
    def __init__(self, generator, discriminator, pose_estimator,
                       pose_penalty_weight = 1000, image_size = (128, 64), **kwargs):
        super(POSE_GAN, self).__init__(generator, discriminator, **kwargs)

        self._pose_penalty_weight = pose_penalty_weight
        self._pose_estimator = pose_estimator
        self._image_size = image_size
        
        self._set_trainable(self._pose_estimator, False)
        self.generator_metric_names = ['struct_128_64', 'struct_64_32', 'struct_32_16', 'gan']
    
    def _compile_generator_loss(self):
        pose_losses = []
        pose_loss_fn = lambda y_true, y_pred, pose_loss: pose_loss
        for i, resolution in enumerate(2 ** np.arange(len(self._generator_output))):
            input_to_estimator = ktf.image.resize_images(self._generator_output[i], self._image_size)
            estimated_pose = self._pose_estimator (input_to_estimator[..., ::-1] / 2) [1]
            #new_size = (self._image_size[0] / (8 * resolution), self._image_size[1] / (8 * resolution))
            #print (new_size)
            reference_pose = self._generator_input[1]
            pose_loss = self._pose_penalty_weight * K.mean((estimated_pose[..., :18] -  reference_pose) ** 2)
            fn = partial(pose_loss_fn, pose_loss  = pose_loss)
            fn.__name__ = 'struct_' + str(i) 
            pose_losses.append(fn)
            
        def pose_loss_fn(y_true, y_pred):
            return sum([pose_losses[i](y_true, y_pred) for i in range(len(pose_losses))], K.zeros((1, )))
        def wasenstein_loss_fn(y_true, y_pred):
            return -K.mean(y_pred)
        def generator_loss_fn(y_true, y_pred):
            return wasenstein_loss_fn(y_true, y_pred) + pose_loss_fn(y_true, y_pred)

        return generator_loss_fn, pose_losses + [wasenstein_loss_fn]
    
    def _compile_discriminator_fake_input(self):
        self._generator_output = self._generator(self._generator_input)
        self._discriminator_fake_input = self._generator_output
        