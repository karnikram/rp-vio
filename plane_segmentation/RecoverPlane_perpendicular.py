from __future__ import division
import os
import time
import math
from data_loader_new import DataLoader
from net import *
from utils import *
import scipy.misc
import random
import math
seed = 8964
tf.set_random_seed(seed)
np.random.seed(seed)

class RecoverPlane(object):
    def __init__(self):
        pass   #do nothing
    # note in python all self.X are class members. And they can be used without declaration.
    def build_train_graph(self):
        opt = self.opt
        self.num_plane = opt.num_plane

        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            self.num_scales)                           # a class defined in data_loader
        with tf.name_scope("data_loading"):
            tgt_image, tgt_depth_stack, tgt_label_stack, intrinsics = loader.load_train_batch() #, tgt2src
            
            tgt_image = self.preprocess_image(tgt_image)              #4*192*320*3


        with tf.name_scope("planeMask_and_planeParam_prediction"):
            pred_param, pred_mask, plane_net_endpoints = \
                plane_pred_net(tgt_image,
                             opt.num_plane,
                             is_training=True)
            new_shape  = (3,3,8)


        with tf.name_scope("mask_color"):
            # Generate random colors
            random.seed(seed)
            colors = random_colors(opt.num_plane)
            for i in range(opt.num_plane):
                colors[i] = [(m*255) for m in colors[i]]

        with tf.name_scope("compute_loss"):
            plane_loss = 0
            depth_loss = 0
            perpendicular_loss = 0
            tgt_image_all = []

            plane_mask_stack_all = []
            color_plane_mask_stack_all = []
            tgt_depth_all = []
            depth_error_all = []
            non_plane_mask_stack_all = []


            for s in range(self.num_scales):
                # Scale the target images, depth, and label for computing loss at the according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                curr_depth_stack = tf.image.resize_area(tgt_depth_stack,
                    [int(opt.img_height / (2 ** s)),int(opt.img_width / (2 ** s))])
                curr_label_stack = tf.image.resize_area(tgt_label_stack,
                    [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])

                # calculate the plane_loss using cross_entropy
                # basically assume all the pixels belong to potential plane masks = 1
                ref_plane_mask = tf.concat([tf.ones(shape=curr_label_stack.get_shape())- curr_label_stack , curr_label_stack], axis=-1)
                plane_loss += opt.plane_weight * \
                              self.compute_plane_reg_loss(pred_mask[s], ref_plane_mask)  # compare with 'ref_plane_mask'
                              
                pred_mask_s = tf.nn.softmax(pred_mask[s])

                # get the unscaled ray, k^(-1)q in Eq.3
                unscaled_ray =  compute_unscaled_ray(curr_tgt_image, intrinsics[:, s, :, :])

                for p in range(opt.num_plane):
                    # the left equation of Eq.3
                    left_plane_eq = compute_plane_equation(curr_tgt_image, pred_param[:,p:p+1,:], unscaled_ray, curr_depth_stack)
                    depth_error = self.compute_depth_error(left_plane_eq, 1.)



                    # extract each plane_mask
                    curr_plane = tf.slice(pred_mask_s,
                                          [0, 0, 0, p],
                                          [-1, -1, -1, 1])

                    # depth_loss compute the variation of depth_error in predicted plane region
                    depth_loss += tf.reduce_mean(depth_error  * curr_plane)

                    if p == 0:
                        depth_error_stack = val2uint8(depth_error,0.3)
                        plane_mask_stack = curr_plane
                    else:
                        depth_error_stack = tf.concat([depth_error_stack,
                                                       val2uint8(depth_error, 0.3) ], axis=-1)

                        plane_mask_stack = tf.concat([plane_mask_stack,
                                                      curr_plane], axis=-1)


                #normlaized depth for visulaization
                norm_tgt_depth = ((curr_depth_stack - tf.reduce_min(curr_depth_stack)) /
                                        (tf.reduce_max(curr_depth_stack) - tf.reduce_min(curr_depth_stack))) * 255

                color_plane_mask = color_mask(self.deprocess_image(curr_tgt_image), pred_mask_s, colors, alpha=0.3)


                #stack all different scale results together
                tgt_image_all.append(curr_tgt_image)
                tgt_depth_all.append(norm_tgt_depth)
                depth_error_all.append(depth_error_stack)
                plane_mask_stack_all.append(plane_mask_stack)
                color_plane_mask_stack_all.append(color_plane_mask)
                non_plane_mask_stack_all.append(pred_mask_s[:,:,:,-1:])

            perpendicular_loss = self.compute_perpendicular_error(pred_param , 1)



            total_loss = depth_loss + plane_loss  + (perpendicular_loss*0.1)



        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate,
                                           opt.beta1,
                                           opt.beta2)

            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)

            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)


        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_param = pred_param
        self.steps_per_epoch = loader.steps_per_epoch # how many step is need to finish one epoch
        self.total_loss = total_loss
        self.plane_loss = plane_loss
        self.depth_loss = depth_loss
        self.perpendicular_loss = perpendicular_loss
        self.tgt_image_all = tgt_image_all
        self.tgt_depth_all = tgt_depth_all
        self.depth_error_all = depth_error_all
        self.plane_mask_stack_all = plane_mask_stack_all
        self.color_plane_mask = color_plane_mask_stack_all
        self.non_plane_mask_stack_all = non_plane_mask_stack_all



    def compute_plane_reg_loss(self, pred_in, ref):
        # eq 5 in paper
        # - max to ensure exp() will not explode to inf
        pred = pred_in - tf.reduce_max(pred_in, axis=-1,keep_dims=True)
        pred_plane_only = pred[:, :, :, :-1]
        # numerical stable implement of
        # plane_mask = tf.reduce_logsumexp(pred_plane_only, axis=-1) - tf.reduce_logsumexp(pred, axis=-1)
        # ensure log() will not explode to -inf
        pred_plane_only_max = tf.reduce_max(pred_plane_only, axis=-1,keep_dims=True)

        plane_mask = tf.reduce_logsumexp(pred_plane_only - pred_plane_only_max, axis=-1,keep_dims=True) + \
                            pred_plane_only_max - tf.reduce_logsumexp(pred, axis=-1,keep_dims=True)
        # combine non plane and plane log(P_pred) together
        non_plane_mask = pred[:, :, :, -1:] - tf.reduce_logsumexp(pred, axis=-1,keep_dims=True)
        curr_pred_mask = tf.concat([non_plane_mask, plane_mask], axis=3)
        
        # caclulate the cross entropy and return
        return  -tf.reduce_mean(tf.reduce_sum(ref * curr_pred_mask, axis=-1) )


    def compute_depth_error(self,proj_homo,proj_depth):

        diff = proj_homo - proj_depth

        l1_diff = tf.reduce_sum(
            tf.abs(diff), axis=-1, keep_dims=True
        )

        depth_error = l1_diff

        return depth_error


    def compute_perpendicular_error(self,  normal_vectors, num):

        # Computes the inter-plane Loss function 

        new_normal_vector = tf.transpose(normal_vectors , perm=[0, 2,1 ]) 


        print(" VECTORIZED Inter Plane LOSS FUNCTION")
        perp_result = [] 
        check= {}
        err = {}  
        result = tf.matmul( new_normal_vector ,normal_vectors ) #shape of tensors [8, 3,3]
        angle_result = tf.acos(result )
        label  = tf.eye( 8, num_columns=3, batch_shape=None)
        zero = tf.zeros([8, 3, 3])
        ones = tf.ones([8,3,3])
        label = tf.where(angle_result > 0.785, zero, ones )
        mid_res = tf.squared_difference(result, label)
        res_error  =tf.reduce_sum((mid_res) , [0,1 ,2])
        perp_loss = 0 
        perp_loss = res_error/8

        return perp_loss



    def collect_summaries(self):   #tf.summary can export model param
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("depth_loss", self.depth_loss)
        tf.summary.scalar("perpendicular_loss", self.perpendicular_loss)
        tf.summary.scalar("plane_loss", self.plane_loss)

        for s in range(self.num_scales):
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]), max_outputs=opt.batch_size)


            tf.summary.image('scale%d_norm_depth_image' % s, \
                             self.tgt_depth_all[s], max_outputs=opt.batch_size)

            tf.summary.image('scale%d_color_masks' % s, \
                             self.color_plane_mask[s], max_outputs=opt.batch_size)

            tf.summary.image('scale%d_non_plane_mask' % s, \
                             self.non_plane_mask_stack_all[s], max_outputs=opt.batch_size)

            for p in range(opt.num_plane):
                tf.summary.image('scale%d_plane_mask_num_%d' % (s, p),
                                 self.plane_mask_stack_all[s][:, :, :, p: p + 1], max_outputs=opt.batch_size)

                tf.summary.image('scale%d_depth_error_%d' % (s, p),
                                 self.depth_error_all[s][:, :, :, p: p + 1], max_outputs=opt.batch_size)

                if s == 0:
                    tf.summary.text("plane_num_%d_n" % (p), tf.as_string(self.pred_param[:, p, :]))


    def train(self, opt):
        # TODO: currently fixed to 4
        self.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()                        #export the result to tensorboard
        self.i = 0 
        with tf.name_scope("parameter_count"):
            # tf.reduce_prod: compute the prodcut of element across dimensions of a tensors
            # parameter_count is the number of params
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])

        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)

        # save the variables of the model and keep the max memory as 10 files
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.chsummaryeckpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step,
                }
                print(str(self.i) +"/800000 ** " +"  code name = train_perpendicular.py"  + " **perpendicular_loss weight  =0.1" + " ** logdir =new_check2" )
                self.i +=1


                if step % opt.summary_freq == 0:
                    fetches["total_loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op

                if step >= opt.max_steps - 100:
                    fetches["masked_res"] = self.color_plane_mask

                results = sess.run(fetches)
                gs = results["global_step"]

                if step >= opt.max_steps - 100:
                    last_res = results["masked_res"]
                    for i in range(opt.batch_size):
                        scipy.misc.imsave(opt.checkpoint_dir + "/res_" + str(step) +"_" + str(i) + ".jpg", last_res[0][i,:,:,:])

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)    # one epoch means all the data in training set is explored once
                                                                          # steps_per_epoch is the len of the data_batch,
                                                                          # gs is the time fetch has been run
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch

                    # print the progress of every 100 iterations
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it total_loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq,\
                                results["total_loss"]\
                               ))

                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)


    # build network for testing
    def build_plane_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height,
                                                self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("plane_predication"):
            pred_param, pred_masks, plane_net_endpoints = plane_pred_net(
                input_mc, num_plane=self.num_plane, is_training=False)
            pred_mask_0 = tf.nn.softmax(pred_masks[0])
        pred_mask = pred_mask_0
        self.inputs = input_uint8
        self.pred_mask = pred_mask
        self.pred_param = pred_param
        self.plane_epts = plane_net_endpoints


    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.    # centeralize

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)


    def setup_inference(self,
                        img_height,
                        img_width,
                        num_plane,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.num_plane = num_plane
        self.batch_size = batch_size
        self.build_plane_test_graph()



    def inference(self, inputs, sess): #, mode='depth'
        fetches = {'pred_param':self.pred_param,
                   'pred_mask':self.pred_mask}

        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

