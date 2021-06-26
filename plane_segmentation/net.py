from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
SCALING = 1.0
BIAS = 0.

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def plane_pred_net(tgt_image, num_plane, is_training=True):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:                          # design the nn architecture for the depth network
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],       #define a conv2d operator with fixed params shown below
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05), # using l2 regularizer with 0.05 weight
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):

            #for slim.conv2d the default padding mode = 'same'
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1') #4*96*160*32
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')     #4*48*80*64
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')     #4*24*40*128
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')     #4*12*20*256
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')    # 4*6*10*256

            with tf.variable_scope('param'):
                cnv6_plane  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6_plane')         # 4*3*5*256
                cnv7_plane  = slim.conv2d(cnv6_plane, 512, [3, 3], stride=2, scope='cnv7_plane')    # 4*2*3*256
                param_pred = slim.conv2d(cnv7_plane, 3*(num_plane), [1, 1], scope='param',          # 4*2*3*3n
                    stride=1, normalizer_fn=None, activation_fn=None)
                param_avg = tf.reduce_mean(param_pred, [1, 2])  #4*3n
                # Empirically we found that scaling by a small constant facilitates training.
                param_final = 0.01 * tf.reshape(param_avg, [-1, (num_plane), 3]) #4*n*3, 2 for n planes in tgt, B*n*num_param

            with tf.variable_scope('mask'):
                upcnv5 = slim.conv2d_transpose(cnv5b, 256, [3, 3], stride=2, scope='upcnv5')
                i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
                icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')                      # 4*12*20*256

                upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')           # 4*24*40*128
                i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
                icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
                segm4  = SCALING * slim.conv2d(icnv4, num_plane + 1,   [3, 3], stride=1,
                        activation_fn=None, normalizer_fn=None, scope='disp4') + BIAS                 # 4*24*40*(1+n)
                segm4_up = tf.image.resize_bilinear(segm4, [np.int(H/4), np.int(W/4)])

                upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')           # 4*48*80*64
                i3_in  = tf.concat([upcnv3, cnv2b, segm4_up], axis=3)
                icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
                segm3  = SCALING * slim.conv2d(icnv3, num_plane + 1,   [3, 3], stride=1,               #4*48*80*(1+n)
                    activation_fn=None, normalizer_fn=None, scope='disp3') + BIAS
                segm3_up = tf.image.resize_bilinear(segm3, [np.int(H/2), np.int(W/2)])

                upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')            # 4*96*160*32
                i2_in  = tf.concat([upcnv2, cnv1b, segm3_up], axis=3)
                icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
                segm2  = SCALING * slim.conv2d(icnv2, num_plane + 1,   [3, 3], stride=1,                 #4*96*160*(n+1)
                    activation_fn=None, normalizer_fn=None, scope='disp2') + BIAS
                segm2_up = tf.image.resize_bilinear(segm2, [H, W])

                upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')            #4*192*320*16
                i1_in  = tf.concat([upcnv1, segm2_up], axis=3)
                icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')                       #4*192*320*(n+1)
                segm1  = SCALING * slim.conv2d(icnv1, num_plane + 1,   [3, 3], stride=1,
                    activation_fn=None, normalizer_fn=None, scope='disp1') + BIAS

            end_points = utils.convert_collection_to_dict(end_points_collection)
            return param_final, [segm1, segm2, segm3, segm4], end_points
