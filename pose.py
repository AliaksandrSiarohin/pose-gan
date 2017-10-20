from conditional_gan import make_generator, make_discriminator, CGAN
from gan.cmd import parser_with_default_args
from gan.train import Trainer

from pose_dataset import PoseHMDataset
from keras.models import load_model, Input


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
    
    gan = CGAN(generator, discriminator, pose_estimator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()
