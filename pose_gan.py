from keras import backend as K
from keras.optimizers import Adam
from wgan_gp import WGAN_GP
from keras.layers import Lambda
from keras.models import Model
from keras.backend import tf as ktf
from keras.layers.merge import Concatenate

class POSE_GAN(WGAN_GP):
    """
        Class for representing pose_gan
    """
    def __init__(self, generator, discriminator, pose_estimator,
                       generator_input_noise, generator_input_pose, discriminator_input,
                       generator_optimizer = Adam(0.0001, beta_1=0., beta_2=0.9),
                       discriminator_optimizer = Adam(0.0001, beta_1=0., beta_2=0.9),
                       cmd_args = None):
        super(POSE_GAN, self).__init__(generator, discriminator, generator_input_noise, 
                        discriminator_input, generator_optimizer, discriminator_optimizer, cmd_args)

        self._generator_input_pose = generator_input_pose
        self._pose_estimator = pose_estimator
        self._pose_penalty_weight = cmd_args.pose_penalty_weight
        
        self._set_trainable(self._pose_estimator, False)
    
    def _compile_generator(self):
        self._set_trainable(self._generator, True)
        self._set_trainable(self._discriminator, False)

        discriminator_output_fake = self._discriminator(self._discriminator_fake_input)

        generator_model = Model(inputs=[self._generator_input, self._generator_input_pose],
                                outputs=[discriminator_output_fake])
        generator_model.compile(optimizer=self._generator_optimizer, loss=self._loss_generator())

        return generator_model
    
    def _compile_discriminator(self):
        self._set_trainable(self._generator, False)
        self._set_trainable(self._discriminator, True)        
        
        disc_in = Concatenate(axis=0)([self._discriminator_input, self._discriminator_fake_input])

        discriminator_model = Model(inputs=[self._discriminator_input, self._generator_input, self._generator_input_pose],
                                    outputs=[self._discriminator(disc_in)])
        discriminator_model.compile(optimizer=self._discriminator_optimizer, loss=self._loss_discriminator())

        return discriminator_model
    
    
    def _loss_generator(self):
        resize_layer = Lambda(lambda x: ktf.image.resize_images(x, (299, 299)))
        def generator_loss(y_true, y_pred):
            estimated_pose = self._pose_estimator (resize_layer(self._discriminator_fake_input))
            true_pose = ktf.cast(self._generator_input_pose, 'float32')
            pose_loss = self._pose_penalty_weight * K.mean((estimated_pose[:, ::2] -  true_pose[:, :, 1]) ** 2)
            pose_loss += self._pose_penalty_weight * K.mean((estimated_pose[:, 1::2] -  true_pose[:, :, 0]) ** 2)
            return -K.mean(y_pred) + pose_loss
        return generator_loss
    
    def compile_models(self):
        self._discriminator_fake_input = self._generator([self._generator_input, self._generator_input_pose])
        return self._compile_generator(), self._compile_discriminator()