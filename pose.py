from conditional_gan import make_generator, make_discriminator, CGAN
from gan.cmd import parser_with_default_args
from gan.train import Trainer

import numpy as np
from pose_dataset import PoseHMDataset
from pose_gan import POSE_GAN
from keras.models import load_model, Input
from gan.layer_utils import LayerNorm
from skimage.transform import resize
import pose_utils
from skimage.io import imsave
from scipy.ndimage.filters import gaussian_filter
def main():
    parser = parser_with_default_args()
    parser.add_argument("--pose_model", default="cao-hpe/pose_estimator.h5", help="Pose estimator")
    parser.add_argument("--pose_generator", default="output/checkpoints_G_S_H/epoch_999_generator.h5", help="Generator of structure")
    parser.add_argument("--pose_penalty_weight", default=1000, type=int, help="Weight of pose penalty")
    
    args = parser.parse_args()
            
    generator = make_generator(Input((128, 64, 18)), Input((128, 64, 3)))
    if args.generator_checkpoint is not  None:
        generator.load_weights(args.generator_checkpoint)
    
    discriminator = make_discriminator(Input((128, 64, 18)), Input((128, 64, 3)))
    if args.discriminator_checkpoint is not None:
        discriminator.load_weights(args.discriminator_checkpoint)
    
    pose_estimator = load_model(args.pose_model)
    pose_generator = load_model(args.pose_generator)    
    
    dataset = PoseHMDataset(args.input_folder, pose_estimator, args.batch_size, (64,), (128, 64))
    
    # poses, mask = pose_generator.predict([np.random.normal(size = (args.batch_size, 64)),
    #                                       np.random.normal(size = (args.batch_size, 64))])
#     poses = pose_generator.predict([np.random.normal(size = (args.batch_size, 64))])
    
    
#     pose_resized = np.array([resize(pose, (128, 64), preserve_range=True)[..., 0:1] for pose in poses])
#     pose_resized = np.array([gaussian_filter(pose, 1) for pose in pose_resized])
#     print (pose_resized.shape)
    
#     batch = dataset.next_generator_sample()
#     pose_ref = batch[0][..., 0:1]
#     from gan.dataset import UGANDataset
#     d = UGANDataset(64, 1)
#     img = np.concatenate([d.display(pose_resized), d.display(pose_ref)], axis=1)
#     print (img.shape)
#     imsave('hm.png', np.squeeze(img))
    
#     def deprocess_array(X):
#         X = X / 2 + 0.5
#         X = X.reshape((X.shape[0], 18, 2))
#         X[...,0] *= 128 - 0.1
#         X[...,1] *= 64 - 0.1
#         return X
#     poses = deprocess_array(poses)
#     poses[mask.reshape((-1, 18, 2)) < 0.5] = -1
#     poses *= 1.2
#     poses = poses.astype(np.int)
#     import keras.backend as K
#     import tensorflow as ktf
#     img_size_init = (int(1.05 * 128), int(1.05 * 64))
#     K._LEARNING_PHASE = ktf.constant(1)
#     batch = dataset.next_generator_sample()
    
#     poses = [resize(pose, img_size_init, preserve_range=True) for pose in poses]
#     poses = (pose_utils.map_to_cord(pose) for pose in poses)
#     poses = [pose_utils.cords_to_map(pose, img_size_init) for pose in poses]
    
#     batch[0] = poses
    
#     batch[0] = np.array([dataset._random_crop(b, b)[0] for b in batch[0]])
    
#     img = dataset.display(generator.predict(batch), batch, row=8, col=8)
#     imsave('cgan_res_hm_h_gt_no_crop.png', img[:,:img.shape[1]/2])

    
    
    gan = CGAN(generator, discriminator, pose_estimator, 
                 custom_objects = {'LayerNorm':LayerNorm}, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()
    
