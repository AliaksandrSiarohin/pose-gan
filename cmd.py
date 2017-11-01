import argparse
from ast import literal_eval as make_tuple


def args():
    """
        Define args that is used in project
    """
    parser = argparse.ArgumentParser(description="Pose guided image generation usign deformable skip layers")
    parser.add_argument("--output_dir", default='output/displayed_samples', help="Directory with generated sample images")
    parser.add_argument("--batch_size", default=4, type=int, help='Size of the batch')
    parser.add_argument("--training_ratio", default=1, type=int,
                        help="The training ratio is the number of discriminator updates per generator update.")

    parser.add_argument("--l1_penalty_weight", default=100, type=float, help='Weight of l1 loss')
    parser.add_argument('--gan_penalty_weight', default=1, type=float, help='Weight of GAN loss')
    parser.add_argument('--tv_penalty_weight', default=1e-4, type=float, help='Weight of total variation loss')
    
    
    parser.add_argument("--number_of_epochs", default=200, type=int, help="Number of training epochs")

    parser.add_argument("--content_loss_layer", default='none', help='Name of content layer (vgg19)'
                                                                     ' e.g. block4_conv1 or none')

    parser.add_argument("--checkpoints_dir", default="output/checkpoints", help="Folder with checkpoints")
    parser.add_argument("--checkpoint_ratio", default=10, type=int, help="Number of epochs between consecutive checkpoints")
    parser.add_argument("--generator_checkpoint", default=None, help="Previosly saved model of generator")
    parser.add_argument("--discriminator_checkpoint", default=None, help="Previosly saved model of discriminator")


    parser.add_argument('--dataset', default='market', choices=['market', 'fasion'],
                        help='Market or fasion')


    parser.add_argument("--display_ratio", default=1, type=int,  help='Number of epochs between ploting')
    parser.add_argument("--start_epoch", default=0, type=int, help='Start epoch for starting from checkpoint')
    parser.add_argument("--pose_estimator", default='pose_estimator.h5',
                            help='Pretrained model for cao pose estimator')

    parser.add_argument("--images_for_test", default=12000, type=int, help="Number of images for testing")

    parser.add_argument("--use_input_pose", default=True, type=int, help='Feed to generator input pose')
    parser.add_argument("--warp_skip", default='none', choices=['none', 'full', 'mask'],
                        help="Type of warping skip layers to use.")

    parser.add_argument("--generated_images_dir", default='output/generated_images',
                        help='Folder with generated images from training dataset')
    parser.add_argument("--generated_images_save_format", default='iog',
                        help='Format of generated images i - input, o - output, g - generated')

    args = parser.parse_args()

    args.images_dir_train = 'data/' + args.dataset + '-dataset/train'
    args.images_dir_test = 'data/' + args.dataset + '-dataset/test'

    args.annotations_file_train = 'data/' + args.dataset + '-annotation-train.csv'
    args.annotations_file_test = 'data/' + args.dataset + '-annotation-test.csv'

    args.pairs_file_train = 'data/' + args.dataset + '-pairs-train.csv'
    args.pairs_file_test = 'data/' + args.dataset + '-pairs-test.csv'

    args.image_size = (128, 64) if args.dataset == 'market' else (256, 256)

    del args.dataset

    return args
