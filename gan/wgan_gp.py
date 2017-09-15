from keras import backend as K
from keras.optimizers import Adam
from gan import GAN

class WGAN_GP(GAN):
    """
        Class for representing WGAN_GP (https://arxiv.org/abs/1704.00028)
    """
    def __init__(self, generator, discriminator,
                       gradient_penalty_weight = 10,
                       **kwargs):
        super(WGAN_GP, self).__init__(generator, discriminator, **kwargs)
        self._gradient_penalty_weight = gradient_penalty_weight
        
        self.generator_metric_names = []
        self.discriminator_metric_names = ['gp_loss_' + str(i) for i in range(len(self._discriminator_input))] + ['true', 'fake']

    
    def _compile_generator_loss(self):
        def generator_wasserstein_loss(y_true, y_pred):
            return -K.mean(y_pred)
        return generator_wasserstein_loss, []
    
    def _compile_discriminator_loss(self):        
        def true_loss(y_true, y_pred):
            y_true = y_pred[:self._batch_size]
            return -K.mean(y_true)
            
        def fake_loss(y_true, y_pred):
            y_fake = y_pred[self._batch_size:]
            return K.mean(y_fake)
                
        real = self._discriminator_input
        fake = self._discriminator_fake_input
        weights = K.random_uniform((self._batch_size, 1, 1, 1))
        averaged_samples = [(weights * r) + ((1 - weights) * f) for r, f in zip(real, fake)]
        
        gp_list = []
        for averaged_samples_part in averaged_samples:
            gradients = K.gradients(K.sum(self._discriminator(averaged_samples)), averaged_samples_part)[0]
            gradients = K.reshape(gradients, (self._batch_size, -1))
            gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis = 1))
            gradient_penalty = self._gradient_penalty_weight * K.square(1 - gradient_l2_norm)
            gp_list.append(K.mean(gradient_penalty))
        
        
        gp_fn_list = [lambda y_true, y_pred: gp_list[i] for i in range(len(gp_list))]
        for i, gp_fn in enumerate(gp_fn_list):
            gp_fn.__name__ = 'gp_loss_' + str(i)
        
        def gp_loss(y_true, y_pred):
            return sum(gp_list, K.zeros((1, ))) 
            
        def discriminator_wasserstein_loss(y_true, y_pred):           
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred) + gp_loss(y_true, y_pred)
        
        return discriminator_wasserstein_loss, gp_fn_list + [true_loss, fake_loss]