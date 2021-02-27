from __future__ import division
import tensorflow as tf
import numpy as np
import os
import random
import colorsys
import time
import scipy.misc
import PIL.Image as pil
import cv2


# from utils_depth_only import *
from RecoverPlane import RecoverPlane

'''
@author -- Fengting Yang 
@modified by Sudarshan

@usage:
    test the train result(param) with depth prediction metric. 

@Output:
   1. plane masks and the visualization, 
   2. pred_depth maps
   3. the statistic metric of depth prediction (see on the terminal)


@parameters:
   1. main parameters coudl be seen in the FLAGS
   2. intrinsics:       The camera intrinsics, note if the image is resized,  please reset the intrinsics correspondingly
'''


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 192, "Image height") 
flags.DEFINE_integer("img_width", 320, "Image width")
flags.DEFINE_integer("num_plane",3, "plane number")
flags.DEFINE_boolean("use_preprocessed", True, 'if use the propocessed data we provided for test' )
flags.DEFINE_string("dataset_dir", '', "Filtered Dataset directory")
flags.DEFINE_string("output_dir", '', "Output directory")
flags.DEFINE_string("gpu", "0", "GPU ID")
flags.DEFINE_string("test_list", 'data_pre_processing/SYNTHIA/tst_100.txt', "Test list")
flags.DEFINE_string("ckpt_file", 'pre_trained_model/synthia_498000', "checkpoint file")
FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

print(FLAGS.gpu)

#intrinsics =  np.array(([[133.185088,0.,160.000000], [ 0., 134.587036,96.000000], [0., 0., 1.]]))
focalLength_x =133.185075
focalLength_y = 134.587036
centerX = 160.000000
centerY =  96.000000

TEST_LIST = str(FLAGS.test_list)
num_test = 100
MAX_DEPTH = 100.
MIN_DEPTH = 0.1

seed = 999


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
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, max_mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == max_mask,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def color_mask(image, pred_masks, colors, alpha=0.5 ):
    '''do iteration to assign the color to the corrosponding mask
       Based on experiment, the first plane will be red, and the second one will be green-blue, when num=2
       and drak blue ,red ,green for num=3
    '''
    N = FLAGS.num_plane
    masked_image = np.copy(image)
    new_colors = [ [ 255,125,255 ] , [255,255,255] , [125,255,0]  ] 
   #  masked_image = image  # change the original color as well, so the model could have the plane color when we visualize it
    max_mask = np.max(pred_masks, axis=-1)
    for i in range(N):
        color = new_colors[i]
        mask = pred_masks[:,:,i]
        masked_image = apply_mask(masked_image, mask, max_mask, color, alpha)

    return masked_image


#**************************************get the predicted plane mask *******************
def thres_mask(pred_masks, num_plane):
    '''
        find each plane's mask using the argmax(prob) approach
    '''
    thres_mask = np.zeros(pred_masks.shape)                 #num_plane + 1 non-plane
    max_mask = np.max(pred_masks, axis=-1)  # here for depth prediction, only plane possiblity is used
    for p in range(num_plane + 1):
        plane_mask = pred_masks[:, :, p]
        # in each channel the region with value 1 corresponding to the plane area in this plane
        thres_mask[:,:,p] = np.where(plane_mask == max_mask, thres_mask[:,:,p]+1., thres_mask[:,:,p])

    return thres_mask

#************************************get depth from plane parameters****************************
def meshgrid(height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = np.matmul(np.ones(shape=[height,1]),np.expand_dims(np.linspace(-1.,1,width),1).T)
  y_t = np.matmul(np.expand_dims(np.linspace(-1.0, 1.0, height), 1),np.ones(shape=np.stack([1, width])))

  x_t = (x_t + 1.0) * 0.5 * (width - 1)
  y_t = (y_t + 1.0) * 0.5 * (height - 1)
  if is_homogeneous:
    ones = np.ones_like(x_t)
    coords =np.stack([x_t, y_t, ones], axis=0)
  else:
    coords = np.stack([x_t, y_t], axis=0)
  return coords

def compute_depth(img, pred_param, num_plane, intrinsics):
    height = img.shape[0]
    width  = img.shape[1]

    # Construct pixel grid coordinates
    pixel_coords = meshgrid(height, width)  # 3*128*416

    cam_coords = np.reshape(pixel_coords, [3, -1])

    unscaled_ray_Q = np.matmul(np.linalg.inv(intrinsics), cam_coords)

    for k in range(num_plane):
        n_div_d = np.expand_dims(pred_param[ k, :], axis=0)
        scale =  1./ (np.matmul(n_div_d, np.matmul(np.linalg.inv(intrinsics), cam_coords)) + 1e-10)


        plane_based_Q = scale * (unscaled_ray_Q )
        plane_based_Q = np.reshape(plane_based_Q, [3, height, width])
        plane_based_Q = np.transpose(plane_based_Q, [1, 2, 0])

        if k == 0:
            plane_depth_stack =  plane_based_Q[:,:,-1:]
        else:
            plane_depth_stack = np.concatenate([plane_depth_stack,
                                         plane_based_Q[ :,:, -1:]], axis=-1)

    return plane_depth_stack

#*********************************compute depth error**************************************
def compute_errors(gt, pred):
    b_empyt = (gt.size == 0)   # if there is no groundtruth available
    b_non_zero = np.all(gt * pred) # if one of them are 0 this will return false
    if b_empyt or not b_non_zero:
        return [-100, -100, -100, -100, -100, -100, -100]   #this tst image will be ignored
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


#*****************************************************************************************************

def main(_):
    with open(TEST_LIST, 'r') as f:
        test_files_list = []
        #depth_file_list = []
        test_files = f.readlines()
        for t in test_files:
            t_split = t[:-1].split()

            if not FLAGS.use_preprocessed:
                # use these two lines only if you preprocessed the dataset from scratch
                test_files_list.append(FLAGS.dataset_dir + '/' +  t_split[-1] )
                #depth_file_list.append(FLAGS.dataset_dir + '/' + t_split[0] +'/'+ t_split[-1] + '_depth.png')
            else:
                # use these two lines if you use our preprocessed dataset
                if t_split[0] == '22': # seq 22 is not available in our preprocessed dataset, see README for more details
                    continue
                test_files_list.append(FLAGS.dataset_dir + '/' + t_split[0] +'/'+ t_split[-1] )
                #depth_file_list.append(FLAGS.dataset_dir + '/' + t_split[0] +'/'+ t_split[-1] + '_depth.png')

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    basename = os.path.basename(FLAGS.ckpt_file)

    # to ensure the consistant color map
    default_top_five_colors = [(0.8, 0.0, 1.0), (0.8, 1.0, 0.0), (0.0, 1.0, 0.4), (1.0, 0.0, 0.0), (0.0, 0.4, 1.0)]
    if FLAGS.num_plane <= 5:
        colors = default_top_five_colors
    else:
        # Generate random colors
        colors = random_colors( FLAGS.num_plane)
        cnt = 0
        for i in default_top_five_colors:
            if i not in colors:
                colors[cnt] = i
                cnt += 1



    planeRecover = RecoverPlane()
    planeRecover.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        num_plane=FLAGS.num_plane
                       )

    rms = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel = np.zeros(num_test, np.float32)
    a1 = np.zeros(num_test, np.float32)
    a2 = np.zeros(num_test, np.float32)
    a3 = np.zeros(num_test, np.float32)

    avg_time = 0.
    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)

        pred_all_masks = []
        pred_all_param = []
        for t in range(0, len(test_files_list), FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files_list)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files_list):
                    break
                fh = open(test_files_list[idx], 'r')
                raw_im =  cv2.imread(test_files_list[idx]) #pil.open(fh)
                try:
                    scaled_im = cv2.resize(raw_im , (FLAGS.img_width, FLAGS.img_height), interpolation = cv2.INTER_AREA)
                    inputs[b] = np.array(scaled_im)

                except:

                    print( str(test_files_list[idx]) + "is not read")

            start_time = time.time()
            pred = planeRecover.inference(inputs, sess)
            cost_time = time.time() - start_time
            avg_time += cost_time
            print("No.%d batch cost_time: %f/img" % (int(t / FLAGS.batch_size), cost_time / FLAGS.batch_size))


            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files_list):
                    break

                color_plane_mask = color_mask(inputs[b], pred['pred_mask'][b, :, :, :], colors, alpha=0.6)
                
                thres_masks = thres_mask(pred['pred_mask'][b, :, :, :],
                                         FLAGS.num_plane)  # this will include non-plane mask at the last channle for depth

                #pred_depth = compute_depth(inputs[b], pred['pred_param'][b, :, :], FLAGS.num_plane, intrinsics)
                #masked_pred_depth = np.zeros([FLAGS.img_height, FLAGS.img_width, 1])
                combined_mask = np.zeros([FLAGS.img_height, FLAGS.img_width, 1])


                for p in range(FLAGS.num_plane):
                    #masked_pred_depth += pred_depth[:, :, p: p + 1] * thres_masks[:, :, p: p + 1]
                    combined_mask += (p + 1) * thres_masks[:, :, p: p + 1]  # for matlab to eval all the plane should start from 1, and the last channle will be 0 as non-plane


                name = test_files_list[idx].split('/')
                if not FLAGS.use_preprocessed:
                    pic_name = name[-3] + '_' + name[-1]
                else:
                    pic_name = name[-2] + '_' + name[-1].replace('.jpg', '.png')

                visual_path = FLAGS.output_dir + '/plane_sgmts_vis/'
                eval_mask_path = FLAGS.output_dir + '/plane_sgmts/'
                modified_eval_mask_path = FLAGS.output_dir + '/plane_sgmts_modfied/'

                if not os.path.exists(visual_path):
                    os.makedirs(visual_path)

                if not os.path.exists(eval_mask_path):
                    os.makedirs(eval_mask_path)
                    os.makedirs(modified_eval_mask_path)

                scipy.misc.imsave(visual_path + pic_name, color_plane_mask)
                #combined_mask = combined_mask
                
                cv2.imwrite(eval_mask_path + pic_name, combined_mask)  # misc will normalize the number to 255 not good
                modified_combined_mask = combined_mask*60
                cv2.imwrite(modified_eval_mask_path + pic_name, modified_combined_mask)
                



if __name__ == '__main__':
    tf.app.run()
