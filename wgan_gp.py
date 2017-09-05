from keras import backend as K
from keras.optimizers import Adam
from gan import GAN

def gradient_penalty_loss(discriminator, averaged_samples, batch_size, gradient_penalty_weight):
    gradients = K.gradients(K.sum(discriminator(averaged_samples)), averaged_samples)[0]
    gradients = K.reshape(gradients, (batch_size, -1))
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis = 1))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

class WGAN_GP(GAN):
    """
        Class for representing WGAN_GP (https://arxiv.org/abs/1704.00028)
    """
    def __init__(self, generator, discriminator,
                       generator_input, discriminator_input,
                       generator_optimizer = Adam(0.0001, beta_1=0., beta_2=0.9),
                       discriminator_optimizer = Adam(0.0001, beta_1=0., beta_2=0.9),
                       cmd_args = None):
        super(WGAN_GP, self).__init__(generator, discriminator, generator_input, 
                        discriminator_input, generator_optimizer, discriminator_optimizer, cmd_args)
        self._gradient_penalty_weight = cmd_args.gradient_penalty_weight
        self._batch_size = cmd_args.batch_size
    
    def _loss_generator(self):
        def generator_wasserstein_loss(y_true, y_pred):
            return -K.mean(y_pred)
        return generator_wasserstein_loss
    
    def _loss_discriminator(self):
        real = self._discriminator_input
        fake = self._discriminator_fake_input
        weights = K.random_uniform((self._batch_size, 1, 1, 1))
        averaged_samples = (weights * real) + ((1 - weights) * fake)
        
        def discriminator_wasserstein_loss(y_true, y_pred):
            y_fake = y_pred[self._batch_size:]
            y_true = y_pred[:self._batch_size]
            return K.mean(y_fake) - K.mean(y_true) + gradient_penalty_loss(self._discriminator,
                                                     averaged_samples, self._batch_size, self._gradient_penalty_weight)
        
        return discriminator_wasserstein_loss
        

