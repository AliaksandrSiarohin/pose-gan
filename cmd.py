import argparse


def parser():
    """
        Define args that is used in project
    """
    parser = argparse.ArgumentParser(description="Pose guided image generation usign deformable skip layers")
    parser.add_argument("--output_dir", default='output/generated_samples', help="Directory with generated sample images")
    parser.add_argument("--batch_size", default=4, type=int, help='Size of the batch')
    parser.add_argument("--training_ratio", default=1, type=int,
                        help="The training ratio is the number of discriminator updates per generator update.")

    parser.add_argument("--l1_penalty_weight", default=100, type=float, help='Weight of gradient penalty loss')
    parser.add_argument("--number_of_epochs", default=200, type=int, help="Number of training epochs")

    parser.add_argument("--checkpoints_dir", default="output/checkpoints", help="Folder with checkpoints")
    parser.add_argument("--checkpoint_ratio", default=10, type=int, help="Number of epochs between consecutive checkpoints")
    parser.add_argument("--generator_checkpoint", default=None, help="Previosly saved model of generator")
    parser.add_argument("--discriminator_checkpoint", default=None, help="Previosly saved model of discriminator")

    parser.add_argument("--images_dir_train", default='../market-dataset/bounding_box_train',
                        help='Folder with real images for training')
    parser.add_argument("--images_dir_test", default='../market-dataset/bounding_box_test',
                        help='Folder with real images for testing')

    parser.add_argument("--annotations_file_train", default='cao-hpe/annotations-train.csv',
                        help='Cordinates annotations for train set')
    parser.add_argument("--annotations_file_test", default='cao-hpe/annotations-test.csv',
                        help='Coordinates annotations for train set')

    parser.add_argument("--pose_hm_dir", default='pose-hm-tmp', help='Folder to store pose heatmaps')
    parser.add_argument("--warp_dir", default='warp-tmp', help='Folder to score warps')

    parser.add_argument("--display_ratio", default=1, type=int,  help='Number of epochs between ploting')
    parser.add_argument("--start_epoch", default=0, type=int, help='Start epoch for starting from checkpoint')
    parser.add_argument("--pose_estimator", default='cao-hpe/pose_estimator.h5',
                            help='Pretrained model for cao pose estimator')
    return parser
