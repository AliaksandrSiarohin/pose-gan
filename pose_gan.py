from keras import backend as K
from gan.wgan_gp import WGAN_GP

class POSE_GAN(WGAN_GP):
    """
        Class for representing pose_gan
    """
    def __init__(self, generator, discriminator, pose_estimator,
                       pose_penalty_weight = 1000, **kwargs):
        super(POSE_GAN, self).__init__(generator, discriminator, **kwargs)

        self._pose_penalty_weight = pose_penalty_weight
        self._pose_estimator = pose_estimator
        
        self._set_trainable(self._pose_estimator, False)
        self.generator_metric_names = ['struct', 'gan']
    
    def _compile_generator_loss(self):
        estimated_pose = self._pose_estimator (self._discriminator_fake_input[0][..., ::-1] / 2) [1]
        pose_loss = self._pose_penalty_weight * K.mean((estimated_pose[..., :18] -  self._generator_input[1]) ** 2)
        def pose_loss_fn(y_true, y_pred):
            return pose_loss
        def wasenstein_loss_fn(y_true, y_pred):
            return -K.mean(y_pred)
        def generator_loss_fn(y_true, y_pred):
            return wasenstein_loss_fn(y_true, y_pred) + pose_loss_fn(y_true, y_pred)

        return generator_loss_fn, [pose_loss_fn, wasenstein_loss_fn]