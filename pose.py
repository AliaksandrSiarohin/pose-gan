from pose_guided_architectures import make_generator, make_discriminator
from gan.cmd import parser_with_default_args
from gan.train import Trainer

import numpy as np
from pose_dataset import PoseHMDataset
from pose_gan import POSE_GAN
from keras.models import load_model, Input
from gan.layer_utils import LayerNorm
    
def main():
    parser = parser_with_default_args()
    parser.add_argument("--pose_model", default="cao-hpe/pose_estimator.h5", help="Pose estimator")
    parser.add_argument("--pose_generator", default="structure_generator.h5", help="Generator of structure")
    parser.add_argument("--pose_penalty_weight", default=1000, type=int, help="Weight of pose penalty")
    
    args = parser.parse_args()
    
    if args.generator_checkpoint is  None:
        generator = make_generator(Input((64,)), Input((16, 8, 18)))
    else:
        geneartor = args.generator_checkpoint
    
    if args.discriminator_checkpoint is None:
        discriminator = make_discriminator(Input((128, 64, 3)))
    else:
        discriminator = args.discriminator_checkpoint
    
    pose_estimator = load_model(args.pose_model)
    pose_generator = load_model(args.pose_generator)    
    
    dataset = PoseHMDataset(args.input_folder, pose_generator, args.batch_size, (64,), (128, 64))
    gan = POSE_GAN(generator, discriminator, pose_estimator, 
                   custom_objects = {'LayerNorm':LayerNorm}, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()
