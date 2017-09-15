from keras import backend as K
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
import numpy as np

class GAN(object):
    """
        Simple gan described in https://arxiv.org/abs/1406.2661
    """

    def __init__(self, generator, discriminator,
                 generator_optimizer=Adam(0.0001, beta_1=0, beta_2=0.9),
                 discriminator_optimizer=Adam(0.0001, beta_1=0, beta_2=0.9), 
                 batch_size = 64, custom_objects = {}, **kwargs):
        
        if type(generator) == str:
            self._generator = load_model(generator, custom_objects = custom_objects)
        else:
            self._generator = generator
            
        if type(discriminator) == str:            
            self._discriminator = load_model(discriminator, custom_objects = custom_objects)
        else:
            self._discriminator = discriminator
               
        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        generator_input = self._generator.input
        discriminator_input = self._discriminator.input        
        
        if type(generator_input) == list:                   
            self._generator_input = generator_input
        else:
            self._generator_input = [generator_input]
            
        if type(discriminator_input) == list:
            self._discriminator_input = discriminator_input
        else:
            self._discriminator_input = [discriminator_input]
            
        self._batch_size = batch_size
        
        self.generator_metric_names = []
        self.discriminator_metric_names = ['true', 'fake']

    def _set_trainable(self, net, trainable):
        for layer in net.layers:
            layer.trainable = trainable
        net.trainable = trainable

    def _compile_generator(self):
        """
            Create Generator model that from noise produce images. It`s trained usign discriminator
        """
        self._set_trainable(self._generator, True)
        self._set_trainable(self._discriminator, False)
        
        discriminator_output_fake = self._discriminator(self._discriminator_fake_input)

        generator_model = Model(inputs=self._generator_input, outputs=discriminator_output_fake)
        loss, metrics = self._compile_generator_loss()
        generator_model.compile(optimizer=self._generator_optimizer, loss=loss, metrics = metrics)

        return generator_model

    def _compile_generator_loss(self):
        """
            Create generator loss and metrics
        """
        
        def generator_crossentrohy_loss(y_true, y_pred):
            return K.mean(K.log(y_pred + 1e-7))
        return generator_crossentrohy_loss, []

    def _compile_discriminator(self):
        """
            Create model that produce discriminator scores from real_data and noise(that will be inputed to generator)
        """
        self._set_trainable(self._generator, False)
        self._set_trainable(self._discriminator, True)        
        
        disc_in = [Concatenate(axis=0)([true, fake])
                   for true, fake in zip(self._discriminator_input, self._discriminator_fake_input)]

        discriminator_model = Model(inputs=self._discriminator_input + self._generator_input,
                                    outputs=self._discriminator(disc_in))
        loss, metrics = self._compile_discriminator_loss()
        discriminator_model.compile(optimizer=self._discriminator_optimizer, loss=loss, metrics = metrics)

        return discriminator_model

    def _compile_discriminator_loss(self):
        """
            Create generator loss and metrics
        """
        def fake_loss(y_true, y_pred):
            return K.mean(K.log(1 - y_pred[self._batch_size:] + 1e-7))
        def true_loss(y_true, y_pred):
            return K.mean(K.log(y_pred[:self._batch_size] + 1e-7))
        def discriminator_crossentrohy_loss(y_true, y_pred):
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred)
        return discriminator_crossentrohy_loss, [true_loss, fake_loss]

    def compile_models(self):
        self._discriminator_fake_input = self._generator(self._generator_input)
        if type(self._discriminator_fake_input) != list:
            self._discriminator_fake_input = [self._discriminator_fake_input]
        return self._compile_generator(), self._compile_discriminator()
    
    def get_generator(self):
        return self._generator
    
    def get_discriminator(self):
        return self._discriminator
    
    def get_losses_as_string(self, generator_losses, discriminator_losses):
        def combine(name_list, losses):
            losses = np.array(losses)            
            if len(losses.shape) == 0:
                losses = losses.reshape((1, ))
            return '; '.join([name + ' = ' + str(loss) for name, loss in zip(name_list, losses)])
        generator_loss_str = combine(['Generator loss'] + self.generator_metric_names, generator_losses)
        discriminator_loss_str = combine(['Disciminator loss'] + self.discriminator_metric_names, discriminator_losses)
        return generator_loss_str, discriminator_loss_str