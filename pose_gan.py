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
                       generator_input_noise, generator_input_pose, 
                       discriminator_input, discriminator_input_pose,
                       generator_optimizer = Adam(0.0001, beta_1=0., beta_2=0.9),
                       discriminator_optimizer = Adam(0.0001, beta_1=0., beta_2=0.9),
                       cmd_args = None):
        super(POSE_GAN, self).__init__(generator, discriminator, generator_input_noise, 
                        discriminator_input, generator_optimizer, discriminator_optimizer, cmd_args)

        self._generator_input_pose = generator_input_pose
        self._discriminator_input_pose = discriminator_input_pose
        self._pose_estimator = pose_estimator
        self._pose_penalty_weight = cmd_args.pose_penalty_weight
        
        self._set_trainable(self._pose_estimator, False)
    
    def _compile_generator(self):
        self._set_trainable(self._generator, True)
        self._set_trainable(self._discriminator, False)

        discriminator_output_fake = self._discriminator([self._discriminator_fake_input])
        generator_model = Model(inputs=[self._generator_input, self._generator_input_pose],
                                outputs=[discriminator_output_fake])
        
        
        generator_model.compile(optimizer=self._generator_optimizer,
                                loss=self._loss_generator(), 
                                metrics = [self.pose_loss_fn, self.generator_wasenstein_loss_fn])

        return generator_model
    
    def _compile_discriminator(self):
        self._set_trainable(self._generator, False)
        self._set_trainable(self._discriminator, True)        
        
        disc_in = Concatenate(axis=0)([self._discriminator_input, self._discriminator_fake_input])
        disc_in_pose = Concatenate(axis=0)([self._discriminator_input_pose, self._generator_input_pose])

        discriminator_model = Model(inputs=[self._discriminator_input, self._discriminator_input_pose,
                                            self._generator_input, self._generator_input_pose],
                                    outputs=[self._discriminator([disc_in])])
        discriminator_model.compile(optimizer=self._discriminator_optimizer, loss=self._loss_discriminator(),
                                   metrics = [self.gp_loss_fn, self.discriminator_wasenstein_loss_fn])

        return discriminator_model
    
    
    def _loss_generator(self):
        estimated_pose = self._pose_estimator (self._discriminator_fake_input[..., ::-1] / 2) [1]
        pose_loss = self._pose_penalty_weight * K.mean((estimated_pose[..., :18] -  self._generator_input_pose) ** 2)
        def pose_loss_fn(y_true, y_pred):
            return pose_loss
        def wasenstein_loss_fn(y_true, y_pred):
            return -K.mean(y_pred)
        def generator_loss_fn(y_true, y_pred):
            return pose_loss_fn(y_true, y_pred) + wasenstein_loss_fn(y_true, y_pred)
        self.pose_loss_fn = pose_loss_fn
        self.generator_wasenstein_loss_fn = wasenstein_loss_fn
        return generator_loss_fn
    
    
    def _loss_discriminator(self):
        real = self._discriminator_input
        fake = self._discriminator_fake_input
        
        weights = K.random_uniform((self._batch_size, 1, 1, 1))
        averaged_samples = (weights * real) + ((1 - weights) * fake)
        averaged_pose = (weights * self._discriminator_input_pose)  + ((1 - weights) * self._generator_input_pose)

        gradients = K.gradients(K.sum(self._discriminator([averaged_samples])), averaged_samples)[0]
        gradients = K.reshape(gradients, (self._batch_size, -1))
        gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis = 1))
        gradient_penalty = self._gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        
        def gp_loss_fn(y_true, y_pred):
            return K.mean(gradient_penalty)
        
        def wasenstein_loss_fn(y_true, y_pred):
            y_fake = y_pred[self._batch_size:]
            y_true = y_pred[:self._batch_size]
            return K.mean(y_fake) - K.mean(y_true)
        
        def discriminator_loss_fn(y_true, y_pred):
            return wasenstein_loss_fn(y_true, y_pred) + gp_loss_fn(y_true, y_pred)
        
        self.gp_loss_fn = gp_loss_fn
        self.discriminator_wasenstein_loss_fn = wasenstein_loss_fn
        
        return discriminator_loss_fn
    
    def compile_models(self):
        self._discriminator_fake_input = self._generator([self._generator_input, self._generator_input_pose])
        return self._compile_generator(), self._compile_discriminator()