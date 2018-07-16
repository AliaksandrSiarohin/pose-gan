from conditional_gan import make_generator, make_discriminator, CGAN
import cmd
from gan.train import Trainer

from pose_dataset import PoseHMDataset


def main():
    args = cmd.args()

    generator = make_generator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type, args.warp_agg,
                               args.use_bg, args.pose_rep_type)
    if args.generator_checkpoint is not None:
        generator.load_weights(args.generator_checkpoint)
    
    discriminator = make_discriminator(args.image_size, args.use_input_pose, args.warp_skip, args.disc_type,
                                       args.warp_agg, args.use_bg, args.pose_rep_type)
    if args.discriminator_checkpoint is not None:
        discriminator.load_weights(args.discriminator_checkpoint)
    
    dataset = PoseHMDataset(test_phase=False, **vars(args))
    
    gan = CGAN(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()
