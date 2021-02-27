from __future__ import division
import os
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None, 
                 batch_size=None, 
                 img_height=None, 
                 img_width=None,
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_scales = num_scales

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)

        # Load the list of training files into queues and shuffle it
        file_list = self.format_file_list(self.dataset_dir, 'train_8000_recent_working')

        print("**************" + "LOADED NEW DATA train_8000_recent_working" + "***********")
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'], 
            seed=seed, 
            shuffle=True)

        cam_paths_queue = tf.train.string_input_producer(
            file_list['cam_file_list'], 
            seed=seed, 
            shuffle=True)

        depth_paths_queue = tf.train.string_input_producer(
            file_list['depth_file_list'],
            seed=seed,
            shuffle=True)

        label_paths_queue = tf.train.string_input_producer(
            file_list['label_file_list'],
            seed=seed,
            shuffle=True)

        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_image(image_contents, channels=3)
        #
        tgt_image = tf.reshape(image_seq, [self.img_height, self.img_width, 3])
	#tgt_image= tf.cast(tgt_image, tf.float32)
	#print( " **** images loaded ******************")

        # Load labels
        label_reader = tf.WholeFileReader()
        _, label_contents = label_reader.read(label_paths_queue)
        label_seq = tf.image.decode_png(label_contents)
        tgt_label = tf.reshape(label_seq, [self.img_height, self.img_width, 1])

        print(" CHECKS PASSED   ")

        # Load depths
        depth_reader = tf.WholeFileReader()
        _, depth_contents = depth_reader.read(depth_paths_queue)
        # image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image_detph = tf.image.decode_png(depth_contents,dtype=tf.uint16 ,channels=1)[:,:, 0]
        tgt_detph = tf.cast(tgt_image_detph, dtype=tf.float32)
        tgt_detph = tf.reshape(tgt_detph, [ self.img_height, self.img_width, 1]) \
                          / tf.constant(100., dtype=tf.float32,shape=[self.img_height, self.img_width, 1])

        # Load camera intrinsics
        print("------------------- intrininsics inncluded ----------" )
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])
        #intrinsics = [ [1169.621094 , 0.000000,  646.295044] , [ 0.000000, 1167.105103 ,489.927032 ] , [ 0.000000 ,0.000000, 1.000000] ]

        # Form training batches
        tgt_image, tgt_detph, tgt_label, intrinsics = \
                tf.train.batch([tgt_image, tgt_detph,tgt_label, intrinsics],
                               batch_size=self.batch_size) #it will upload 4 batch from the dataset

        # Data augmentation
        tgt_image, tgt_detph, tgt_label, intrinsics = self.data_augmentation(
            tgt_image, tgt_detph, tgt_label,intrinsics, self.img_height, self.img_width)

        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)

        return tgt_image, tgt_detph, tgt_label, intrinsics

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, depth, label, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, depth, label, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            depth = tf.image.resize_area(depth, [out_h, out_w])
            label = tf.image.resize_area(label, [out_h, out_w])
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, depth, label, intrinsics

        # Random cropping
        def random_cropping(im, depth, label, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]  #the scale of in_h and out_h can be different
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]  # because of the scaling process runs before it
            im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
            depth = tf.image.crop_to_bounding_box(depth, offset_y, offset_x, out_h, out_w)
            label = tf.image.crop_to_bounding_box(label, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, depth, label, intrinsics
        im, depth, label, intrinsics = random_scaling(im, depth,label, intrinsics)
        im, depth, label, intrinsics = random_cropping(im, depth, label, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        return im,depth,label, intrinsics

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        #print(image_file_list)

        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]

        depth_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_depth.png') for i in range(len(frames))]

        label_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_label.png') for i in range(len(frames))]

        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['depth_file_list'] = depth_file_list
        all_list['label_file_list']= label_file_list
        return all_list


    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
