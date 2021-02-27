# Author: Sudarshan

import sys
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import os
import argparse

try:
    from cv2 import imread, imwrite
except ImportError:


    from skimage.io import imread, imsave
    imwrite = imsave


from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def CRF_act(  fn_im ,fn_anno , fn_output , fn_output2 ):

  anno_rgb = imread(fn_anno).astype(np.uint8)
  anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

  colors_not_used, labels = np.unique(anno_lbl, return_inverse=True)

  img = imread(fn_im)
  img = cv2.resize(img , (320,192), interpolation =cv2.INTER_AREA)

  n_labels   = 4

  colorize = np.empty((len(colors_not_used), 3), np.uint8)
  colorize[:,0] = (colors_not_used & 0x0000FF)
  colorize[:,1] = (colors_not_used & 0x0000FF) >> 8
  colorize[:,2] = (colors_not_used & 0x0000FF) >> 8


  d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)


  U = unary_from_labels(labels, n_labels, gt_prob=0.8, zero_unsure=False)

  d.setUnaryEnergy(U)


  feats = create_pairwise_gaussian(sdims=(5, 5), shape=img.shape[:2])
  d.addPairwiseEnergy(feats, compat=2,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

  feats = create_pairwise_bilateral(sdims=(100, 100), schan=(13, 13, 13),
                                      img=img, chdim=2)

  d.addPairwiseEnergy(feats, compat=7,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


  Q, tmp1, tmp2 = d.startInference()
  for i in range(6):
      print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
      d.stepInference(Q, tmp1, tmp2)
  MAP = np.argmax(Q, axis=0)
  MAP = colorize[MAP,:]
  
  MAP = MAP.reshape(img.shape)
  MAP = cv2.morphologyEx(MAP, cv2.MORPH_CLOSE, kernel)

  print(np.shape(MAP))
  imwrite(fn_output, MAP)

  for rows in range(np.shape(MAP)[0]):
  	for cols in range(np.shape(MAP)[1]):

  		if (MAP[rows][cols][0]  > 0):
  			MAP[rows][cols] =255

  		if (MAP[rows][cols][0]  == 0):
  			MAP[rows][cols] =0

  imwrite(fn_output2, MAP)
  #imwrite(fn_output, MAP.reshape(img.shape))


kernel = (10,10)

parser = argparse.ArgumentParser()
parser.add_argument('rgb_image_dir', help='path to original rgb images')
parser.add_argument('labels_dir', help='path to plane_sgmts_modified output dir from planerecover')
parser.add_argument('output_dir', help='path to output refined masks')

args = parser.parse_args()

RGB_image_dir = args.rgb_image_dir
labels_dir = args.labels_dir
output_dir = args.output_dir
output_dir2= "/home/sudarshan/planerecover/CRF/Results_advio_single"

images = sorted(os.listdir(RGB_image_dir))
labels = sorted(os.listdir(labels_dir))

k = 0
for i, j in zip(images ,  labels):
	k +=1

	if ( k >8948):
  
		  image_name =  i #"img" + str(i) +".jpg"
		  image_path = os.path.join(RGB_image_dir, image_name)

		  labels_name = j #"OpenLORIS_images_img" + str(i) +".png" 
		  labels_path = os.path.join(labels_dir, labels_name)

		  output_name = "img" + str(i) + "_crf_res" +".png"
		  output_path = os.path.join(output_dir, output_name)
		  output_path2 = os.path.join(output_dir2, output_name)


		  CRF_act(image_path , labels_path, output_path, output_path2)
		  

		  print( "***********" + str(image_name) +"/" + "****** " + str(labels_name) + "*****" ) 
