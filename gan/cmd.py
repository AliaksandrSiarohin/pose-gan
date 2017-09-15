import argparse

def parser_with_default_args():
    """
        Define args that is used in default wgan_gp, you can add other args in client.
    """
    parser = argparse.ArgumentParser(description="Improved Wasserstein GAN implementation for Keras.")
    parser.add_argument("--output_dir", default='output/generated_samples', help="Directory with generated sample images")
    parser.add_argument("--batch_size", default=64, type=int, help='Size of the batch')
    parser.add_argument("--training_ratio", default=5, type=int,
                        help="The training ratio is the number of discriminator updates per generator update." + 
                        "The paper uses 5")
    parser.add_argument("--gradient_penalty_weight", default=10, type=float, help='Weight of gradient penalty loss')
    parser.add_argument("--number_of_epochs", default=100, type=int, help="Number of training epochs")
    
    parser.add_argument("--checkpoints_dir", default="output/checkpoints", help="Folder with checkpoints")
    parser.add_argument("--checkpoint_ratio", default=10, type=int, help="Number of epochs between consecutive checkpoints")    
    parser.add_argument("--generator_checkpoint", default=None, help="Previosly saved model of generator")
    parser.add_argument("--discriminator_checkpoint", default=None, help="Previosly saved model of discriminator")
    
    parser.add_argument("--input_folder", default='../market-dataset/bounding_box_train', help='Folder with real images for training')

    parser.add_argument("--display_ratio", default=1, type=int,  help='Number of epochs between ploting')
    parser.add_argument("--start_epoch", default=0, type=int, help='Start epoch for starting from checkpoint')
    
    return parser