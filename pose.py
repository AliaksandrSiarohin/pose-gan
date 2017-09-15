from pose_guided_architectures import make_generator, make_discriminator
from gan.cmd import parser_with_default_args
from gan.train import Trainer

import numpy as np
from pose_dataset import PoseHMDataset
from pose_gan import POSE_GAN
from keras.models import load_model, Input
    
def main():
    generator = make_generator(Input((64,)), Input((16, 8, 18)))
    discriminator = make_discriminator(Input((128, 64, 3)))    
    
    parser = parser_with_default_args()
    parser.add_argument("--pose_model", default="cao-hpe/pose_estimator.h5", help="InceptionV3 based pose estimator")
    parser.add_argument("--pose_penalty_weight", default=1000, type=int, help="Weight of pose penalty")
    parser.add_argument("--pose_folder", default='cao-hpe/annotations', help='Folder pose annotations')
    
    args = parser.parse_args()
    pose_estimator = load_model(args.pose_model)
    
    
    dataset = PoseHMDataset(args.input_folder, args.pose_folder, args.batch_size, (64,), (128, 64))
    gan = POSE_GAN(generator, discriminator, pose_estimator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()