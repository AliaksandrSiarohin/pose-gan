# Deformable GANs for Pose-based Human Image Generation.
### Requirment
* python2
* Numpy
* Scipy
* Skimage
* Pandas
* Tensorflow
* Keras
* tqdm 

### Training
In orger to train a model:
1. Create folder market-dataset with 2 subfolder (test and train). Put the test images in test images in test/ and train images in train/.
2. Download pose estimator (conversion of this https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) [pose_estimator.h5](https://yadi.sk/d/blgmGpDi3PjXvK). Launch ```python compute_cordinates.py.``` It will compute human keypoints.
3. Create pairs dataset with ```python create_pairs_dataset.py```. It define pairs for training.
4. Run ```python train.py``` (see list of parameters in cmd.py)

### Testing
0. Download checkpoints (https://yadi.sk/d/dxVvYxBw3QuUT9).
1. Run ```python test.py --generator_checkpoint path/to/generator/checkpoint``` (and same parameters as in train.py). It generate images and compute inception score, SSIM score and their masked versions.
2. To compute ssd_score. Download pretrained on VOC 300x300 model from https://github.com/weiliu89/caffe/tree/ssd. Put it in the ssd_score forlder. Run ```python compute_ssd_score.py --input_dir path/to/generated/images --img_index 2```
