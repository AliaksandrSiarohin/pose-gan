from small_res_architectures import LayerNorm

from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, Activation, Input, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Add, Concatenate
from keras.layers.pooling import AveragePooling2D
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.backend import tf as ktf
import numpy as np

from gan.layer_utils import LayerNorm, resblock

# class PoseMapFromCordinatesLayer(Layer):
#     def __init__(self, map_size, point_size, **kwargs):
#         self.map_size = map_size
#         self.point_size = point_size
#         assert point_size[0] % 2 == 1, "Point size should be odd and square"
#         super(PoseMapFromCordinatesLayer, self).__init__(**kwargs)
    
#     def build(self, input_shape):
#         point_expand_filter = np.zeros([self.point_size[0], self.point_size[1],
#                                         input_shape[1], input_shape[1]], dtype='float32')        
#         for i in range(point_expand_filter.shape[-1]):
#             point_expand_filter[:, :, i, i] = 1         
#         self.point_expand_filter = ktf.constant(point_expand_filter)
        
        
#         # After valid convolution size of the tensor with ajusted size
#         # will be equal to map_size 
#         self.adjusted_map_size = (self.map_size[0] + self.point_size[0] - 1,
#                                   self.map_size[1] + self.point_size[1] - 1)
        
#         self.index_adjustment = ([(self.point_size[0] - 1) / 2, (self.point_size[1] - 1) / 2] *
#                                                                ktf.ones([2, ], dtype='int32'))
        
    
#     def call(self, x, mask=None):
#         #shape of x (batch_size, number_of_keypoints, 2)
#         def index_pack_to_map(pack):
#             #shape of the pach (number_of_keypoints, 2)
#             def index_to_map(index):
#                 """
#                     Create map with one in given index if it`s not outsize,
#                     all other elements is zeros.
#                 """
#                 index = index + self.index_adjustment
                
                
#                 x_outside = ktf.logical_or(index[0] >= self.adjusted_map_size[0], index[0] < 0)
#                 y_outside = ktf.logical_or(index[1] >= self.adjusted_map_size[1], index[1] < 0)
#                 some_outside = ktf.logical_or(x_outside, y_outside)
                
#                 indices = ktf.expand_dims(index, axis = 0)
#                 updates = ktf.ones((1, ))
#                 shape = ktf.constant(self.adjusted_map_size)
                
#                 res_map = ktf.cond(some_outside, lambda:ktf.zeros(self.adjusted_map_size),
#                                                  lambda:ktf.scatter_nd(indices, updates, shape))
                
#                 return res_map
            
            
#             return ktf.map_fn(index_to_map, pack, dtype='float32', back_prop=False)

#         out = ktf.map_fn(index_pack_to_map, x, dtype='float32', back_prop=False)
        
#         out = ktf.transpose(out, perm=[0, 2, 3, 1])
#         #print (out.shape)
#         #print (self.point_expand_filter.shape)
#         # To increase size of the blob with ones, convolves result map with kernel of ones
#         out = ktf.nn.conv2d(out, self.point_expand_filter, strides=(1, 1, 1, 1), padding='VALID')
       
#         return out
    
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.map_size[0], self.map_size[1], input_shape[1]) 
    
#     def get_config(self):
#         config = {"map_size": self.map_size, "point_size": self.point_size}
#         base_config = super(PoseMapFromCordinatesLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))



def make_generator(noise, pose):
    y_pose_16_8 = resblock(pose, (3, 3), 'SAME', 256)
    y_pose_32_16 = resblock(y_pose_16_8, (3, 3), 'UP', 128)
    y_pose_64_32 = resblock(y_pose_32_16, (3, 3), 'UP', 64)
    y_pose_128_64 = resblock(y_pose_64_32, (3, 3), 'UP', 64)
    
    y = Dense(256 * 8 * 4) (noise)
    y = Reshape((8, 4, 256)) (y)
    
    y = resblock(y, (3, 3), 'UP', 256)
    y = Concatenate(axis=-1) ([y, y_pose_16_8])
    y = resblock(y, (3, 3), 'UP', 128)
    y = Concatenate(axis=-1) ([y, y_pose_32_16])
    y = resblock(y, (3, 3), 'UP', 64)
    y = Concatenate(axis=-1) ([y, y_pose_64_32])
    y = resblock(y, (3, 3), 'UP', 64)
    y = Concatenate(axis=-1) ([y, y_pose_128_64])    
  
    y = resblock(y, (3, 3), 'SAME', 64)
    
    y = BatchNormalization(axis=-1) (y)
    y = Activation('relu') (y)
    y = Conv2D(3, (3, 3), kernel_initializer='he_uniform', use_bias = False, 
                      padding='same', activation='tanh')(y)
    return Model(inputs=[noise, pose], outputs=[y]) 


def make_discriminator(image):
    #y_pose_16_8 = resblock(pose, (3, 3), 'SAME', 256, LayerNorm)
    #y_pose_32_16 = resblock(y_pose_16_8, (3, 3), 'UP', 128, LayerNorm)
    #y_pose_64_32 = resblock(y_pose_32_16, (3, 3), 'UP', 64, LayerNorm)
    
    y = Conv2D(32, (3, 3), kernel_initializer='he_uniform',
                      use_bias = True, padding='same') (image)
    
    y = resblock(y, (3, 3), 'DOWN', 64, LayerNorm)
    #y = Concatenate(axis=-1) ([y, y_pose_64_32])
    y = resblock(y, (3, 3), 'DOWN', 128, LayerNorm)
    #y = Concatenate(axis=-1) ([y, y_pose_32_16])
    y = resblock(y, (3, 3), 'DOWN', 256, LayerNorm)
    #y = Concatenate(axis=-1) ([y, y_pose_16_8])
    y = resblock(y, (3, 3), 'DOWN', 512, LayerNorm)
    
    y = Flatten()(y)
    y = Dense(1, use_bias = False)(y)
    return Model(inputs=[image], outputs=[y])
