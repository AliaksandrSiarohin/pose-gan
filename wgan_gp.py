from keras import backend as K
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from gan import GAN

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(discriminator, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(discriminator(averaged_samples)), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

class WGAN_GP(GAN):
    """
        Class for representing WGAN_GP (https://arxiv.org/abs/1704.00028)
    """
    def __init__(self, generator, discriminator,
                       generator_input, discriminator_input,
                       generator_optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9),
                       discriminator_optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9),
                       cmd_args = None):
        super(WGAN_GP, self).__init__(generator, discriminator, generator_input, 
                        discriminator_input, generator_optimizer, discriminator_optimizer, cmd_args)
        self._gradient_penalty_weight = cmd_args.gradient_penalty_weight
        self._batch_size = cmd_args.batch_size
    
    def _loss_generator(self):
        return wasserstein_loss
    
    def _loss_discriminator(self):
        real = self._discriminator_input[:self._batch_size]
        fake = self._discriminator_input[self._batch_size:]
        weights = K.random_uniform((self._batch_size, 1, 1, 1))
        averaged_samples = (weights * real) + ((1 - weights) * fake)
        
        def discriminator_loss(y_true, y_pred):
            y_true = 2 * (y_true - 0.5)
            return wasserstein_loss(y_true, y_pred) + gradient_penalty_loss(self._discriminator,
                                                           averaged_samples, self._gradient_penalty_weight)
        
        return discriminator_loss
        

