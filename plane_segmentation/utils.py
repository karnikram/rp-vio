from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import colorsys


# ****************************colorful mask part*************************************
# Usage: apply different color to each plane
#        the plane determination is based on the plane_threshold = 0.5 now
#        and the area without additional color are belong to non-plane
#
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(float(i) / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, max_mask, color, alpha=0.4):
    """Apply the given mask to the image.
       alpha here means the alpha channel value
    """
    b,h,w,_ = image.get_shape().as_list()
    res_mask = tf.concat([mask, mask, mask], axis=-1)  # be consistent with color channel number
    ref_max_mask = tf.concat([max_mask,max_mask,max_mask],axis=-1)
    b_equal = tf.equal(res_mask, ref_max_mask)

    alpha_img = tf.where(b_equal, tf.cast(tf.scalar_mul(1 - alpha, tf.cast(image, tf.float32)),
                                            tf.uint8), image)

    alpha_color = alpha * np.array(color)
    ref_color = np.tile(alpha_color.astype(int), (b, h, w, 1))
    ref_color = tf.constant(ref_color, dtype=tf.uint8)

    res_image = tf.where(b_equal, ref_color + alpha_img, alpha_img)

    return res_image


def color_mask(image, pred_mask_s, colors, alpha=0.4 ):
    '''do iteration to assign the color to the corrosponding mask
       Based on experiment, the first plane will be red, and the second one will be green-blue, when num=2
       and drak blue ,red ,green for num=3
    '''
    N = len(colors)
    masked_image = image
    for i in range(N):
        color = colors[i]

        mask = tf.expand_dims(pred_mask_s[:, :, :, i],axis=-1)
        max_mask = tf.reduce_max(pred_mask_s, axis=-1,keep_dims=True)
        masked_image = apply_mask(masked_image, mask, max_mask, color, alpha)

    return masked_image

#*********************************************************************************************


def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0])) #tf.linspace(start,stop,num)
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords


def compute_depth(img, pred_param, num_plane, intrinsics):
    batch, height, width, _ = img.get_shape().as_list()
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)  # 3*128*416

    cam_coords = tf.reshape(pixel_coords, [batch, 3, -1])

    unscaled_ray_Q = tf.matmul(tf.matrix_inverse(intrinsics), cam_coords)

    for k in range(num_plane):
        n_div_d = tf.expand_dims(pred_param[:, k, :], axis=1)
        scale =   tf.matmul(n_div_d, tf.matmul(tf.matrix_inverse(intrinsics), cam_coords))

        plane_based_Q = scale * 1./ (unscaled_ray_Q + 1e-10)
        plane_based_Q = tf.reshape(plane_based_Q, [batch, 3, height, width])
        plane_based_Q = tf.transpose(plane_based_Q, perm=[0, 2, 3, 1])

        if k == 0:
            plane_inv_depth_stack =  plane_based_Q[:,:,:,-1:]
        else:
            plane_inv_depth_stack = tf.concat([plane_inv_depth_stack,
                                         plane_based_Q[:, :,:, -1:]], axis=-1)

    return plane_inv_depth_stack

def compute_unscaled_ray(img, intrinsics):
    batch, height, width, _ = img.get_shape().as_list()
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)  # 3*128*416

    cam_coords = tf.reshape(pixel_coords, [batch, 3, -1])

    unscaled_ray_Q = tf.matmul(tf.matrix_inverse(intrinsics), cam_coords)

    return unscaled_ray_Q

def compute_plane_equation(img, pred_param, ray, depth):

    batch, height, width, _ = img.get_shape().as_list()
    n_time_ray = tf.matmul(pred_param,ray)
    n_time_ray = tf.reshape(n_time_ray, [batch, 1, height, width])
    n_time_ray = tf.transpose(n_time_ray, perm=[0, 2, 3, 1])

    left_eq = n_time_ray * depth

    return  left_eq

def val2uint8(mat,maxVal):
    maxVal_mat = tf.constant(maxVal,tf.float32,mat.get_shape())
    mat_vis = tf.where(tf.greater(mat,maxVal_mat), maxVal_mat, mat)
    return tf.cast(mat_vis * 255. / maxVal, tf.uint8)
