from keras import backend as K
from keras.layers.merge import _Merge
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

    def _set_trainable(self, net, trainable):
        for layer in net.layers:
            layer.trainable = trainable
        net.trainable = trainable

    def _compile_generator(self):
        self._set_trainable(self._generator, True)
        self._set_trainable(self._discriminator, False)

        generator_output = self._generator(self._generator_input)
        discriminator_output_fake = self._discriminator(generator_output)

        generator_model = Model(inputs=[self._generator_input], outputs=[discriminator_output_fake])
        generator_model.compile(optimizer=self._generator_optimizer, loss=self._loss_generator())

        return generator_model

    def _loss_generator(self):
        return 'binary_crossentropy'

    def _compile_discriminator(self):
        self._set_trainable(self._generator, False)
        self._set_trainable(self._discriminator, True)

        discriminator_model = Model(inputs=[self._discriminator_input],
                                    outputs=[self._discriminator(self._discriminator_input)])
        discriminator_model.compile(optimizer=self._discriminator_optimizer, loss=self._loss_discriminator())

        return discriminator_model

    def _loss_discriminator(self):
        return 'binary_crossentropy'

    def compile_models(self):
        return self._compile_generator(), self._compile_discriminator()
