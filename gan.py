from keras import backend as K
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.models import Model, Sequential

class GAN(object):
    """
        Simple gan described in https://arxiv.org/abs/1406.2661
    """

    def __init__(self, generator, discriminator,
                 generator_input, discriminator_input,
                 generator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                 discriminator_optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                 cmd_args=None):
        self._generator = generator
        self._discriminator = discriminator
               
        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        
        self._generator_input = generator_input
        self._discriminator_input = discriminator_input
        self._batch_size = cmd_args.batch_size

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

        generator_model = Model(inputs=[self._generator_input], outputs=[discriminator_output_fake])
        generator_model.compile(optimizer=self._generator_optimizer, loss=self._loss_generator())

        return generator_model

    def _loss_generator(self):
        def generator_crossentrohy_loss(y_true, y_pred):
            return K.mean(K.log(y_pred + 1e-7))
        return generator_crossentrohy_loss

    def _compile_discriminator(self):
        """
            Create model that produce discriminator scores from real_data and noise(that will be inputed to generator)
        """
        self._set_trainable(self._generator, False)
        self._set_trainable(self._discriminator, True)        
        
        disc_in = Concatenate(axis=0)([self._discriminator_input, self._discriminator_fake_input])

        discriminator_model = Model(inputs=[self._discriminator_input, self._generator_input],
                                    outputs=[self._discriminator(disc_in)])
        discriminator_model.compile(optimizer=self._discriminator_optimizer, loss=self._loss_discriminator())

        return discriminator_model

    def _loss_discriminator(self):
        def discriminator_crossentrohy_loss(y_true, y_pred):
            return K.mean(K.log(y_pred[:self._batch_size] + 1e-7)) + K.mean(K.log(1 - y_pred[self._batch_size:] + 1e-7))
        return discriminator_crossentrohy_loss

    def compile_models(self):
        self._discriminator_fake_input = self._generator(self._generator_input)
        return self._compile_generator(), self._compile_discriminator()
