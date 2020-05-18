# -*- coding=utf-8 -*-
from util.image_preprocessing_util import *
from util.network_util import *
from util.loss_uitl import *
from util.metric_util import *
from network import *
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import  perceptual_loss as pl


############################
#configuration
############################


data_root = 'msra10k_0.3/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
version = 'sc_v64/'
log_dir = "log_" + version

fix_size = 128
latent_varibale_length = 200
batch_size = 16
epcho = 1000
learning_rate = 1e-3
weight_decay = 1e-4
#weight_decay = 0
kp = 0.9
one_epoche_iterations = 1000
#one_epoche_iterations = 400
fuse_channals = 12
compress_rate = 0.5
growth_rate = 24
layer = 6

beta1 = 0.5
beta2 = 0.99
w1 = 0.001
#w1 = 0.01
w2 = 0.001
w3 = 0.1
w4 = 1e-7


def load_batch_trainning_samples(train_images,
                                 train_masks,
                                 train_shapes,
                                 other_images,
                                 other_shapes,
                                 fix_size,
                                 batch_size=32,
                                 latent_varibale_length = 100):

    n_train_samples,_,_= np.shape(train_shapes)
    n_other_samples,_,_ = np.shape(other_shapes)

    # n_train_samples = 1
    # n_other_samples = 1

    np.random.seed()
    train_indexs = np.random.random_integers(low=0,high=n_train_samples-1,size=batch_size // 2)
    other_indexs = np.random.random_integers(low=0,high=n_other_samples-1,size=batch_size // 2)

    imgs = []
    masks = []
    other_imgs = []
    codes = []

    for i in train_indexs:

        shape = train_shapes[i, :, :]
        ht, wd, ch = shape[0, :]

        img = train_images[i,:,:,:][0:ht,0:wd,:]
        mask = train_masks[i,:,:,:][0:ht,0:wd,:]

        img_p, mask_p = get_one_preprocessed_img_with_mask(img, mask)
        img_p, mask_p = img_normalize_with_mask(img_p, mask_p, fix_size, fix_size)

        #img_p, mask_p = img_normalize_with_mask(img, mask, fix_size, fix_size)
        np.random.seed(i)
        code = np.random.normal(0,1,size=latent_varibale_length)


        imgs.append(img_p)
        masks.append(mask_p)
        codes.append(code)


    for j in other_indexs:

        shape = other_shapes[j, :, :]
        ht, wd, ch = shape[0, :]

        img = other_images[j, :, :, :][0:ht, 0:wd, :]
        # #cv2.imshow('img', img)
        img_p = get_one_preprocessed_img(img)
        # #cv2.imshow('img_p', img_p)
        img_p = img_normalize(img_p,fix_size,fix_size)
        # #cv2.waitKey()
        #img_p = img_normalize(img, fix_size, fix_size)

        np.random.seed(j)
        code = np.random.normal(0, 1, size=latent_varibale_length)

        other_imgs.append(img_p)
        codes.append(code)

    return imgs,masks,other_imgs,codes


def load_one_test_samples(images,
                           masks,
                           shapes,
                           fix_size,
                           latent_varibale_length=100):

    n_samples,_,_ = np.shape(shapes)


    for i in range(n_samples):

        shape = shapes[i, :, :]
        ht, wd, ch = shape[0, :]

        img = images[i, :, :, :][0:ht, 0:wd, :]
        mask = masks[i, :, :, :][0:ht, 0:wd, :]

        img_p, mask_p = img_normalize_with_mask(img, mask, fix_size, fix_size)

        img_p = np.expand_dims(img_p,axis=0)
        mask_p = np.expand_dims(mask_p,axis=0)

        np.random.seed(i)
        code = np.random.normal(0, 1, size=latent_varibale_length)
        codes = [code,code]

        yield img_p,mask_p,img_p,codes



################################
# define graph
################################

graph = tf.Graph()
with graph.as_default():


    ###############################
    # network input
    ###############################


    x_l = tf.placeholder(tf.float32, shape=[None, fix_size,fix_size,3],name="xl")
    x_u = tf.placeholder(tf.float32, shape=[None, fix_size,fix_size,3],name="xu")
    c = tf.placeholder(tf.float32, shape=[None,latent_varibale_length ],name="c")
    tf.summary.histogram('c',c)

    c_label ,c_unlabel = tf.split(c,num_or_size_splits=2,axis=0)

    x = tf.concat([x_l,x_u],axis=0)
    tf.summary.image("x", x, 4)

    y = tf.placeholder(tf.float32, shape=[None, fix_size,fix_size,1],name="y")
    y_norm = y * 2 - 1

    tf.summary.image("y", y,4)

    #z = tf.placeholder(tf.float32, shape=[None, fix_size,fix_size,1],name="z")
    lr = tf.placeholder(tf.float32, shape=[],name='lr')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    is_training = tf.placeholder("bool", shape=[],name='is_training')

    ###############################
    # network construct
    ###############################


    ################### x->y_hat  G_A ###################

    y_logits = Generator(x,
                         keep_prob,
                         layer=layer,
                         scope='G_A',
                         fuse_channals = fuse_channals,
                         output_channel=1,
                         growth_rate=growth_rate,
                         compress_rate=compress_rate,
                         is_training=is_training,
                         reuse=False,
                         dilated_rates=[1, 1, 1, 1, 1]
                         )


    print('y_logits',y_logits)

    y_logits_labeled, y_logits_unlabeled = tf.split(y_logits, num_or_size_splits=2, axis=1)

    y_logits_final = y_logits[0]
    y_logits_labeled_final = y_logits_labeled[0]
    y_logits_unlabeled_final = y_logits_unlabeled[0]

    y_out_labeled_final = tf.tanh(y_logits_labeled_final)
    y_out_unlabeled_final = tf.tanh(y_logits_unlabeled_final)
    y_out_final = tf.tanh(y_logits_final)

    y_hat = tf.sigmoid(y_logits)

    y_hat_labeled,y_hat_unlabeled = tf.split(y_hat,num_or_size_splits=2,axis=1)

    y_hat_labeled_final = y_hat_labeled[0]
    y_hat_unlabeled_final = y_hat_unlabeled[0]

    tf.identity(y_hat_labeled_final,name='y_hat_labeled_final')
    tf.identity(y_hat_unlabeled_final,name='y_hat_unlabeled_final')

    tf.summary.image('y_hat_labeled_final', y_hat_labeled_final)
    tf.summary.image('y_hat_unlabeled_final', y_hat_unlabeled_final)

    print('y_hat_labeled',y_hat_labeled)
    print('y_hat_unlabeled',y_hat_unlabeled)

    ################### y -> x_hat  G_B ###################



    x_hat_logits = Generator_DC_128(y_norm,
                                    c_label,
                                    base_n_channel=64,
                                    output_channel=3,
                                    keep_prob=keep_prob,
                                    scope='G_B',
                                    is_training=is_training,
                                    reuse=False)


    x_hat_logits_final = x_hat_logits
    x_hat_out_final = tf.tanh(x_hat_logits_final)



    tf.summary.image('x_hat_out_final',x_hat_out_final)


    ################### y_hat -> x_cycle G_B  ###################


    x_cycle_logits = Generator_DC_128(y_out_final,
                                    c,
                                    base_n_channel=64,
                                    output_channel=3,
                                    keep_prob=keep_prob,
                                    scope='G_B',
                                    is_training=is_training,
                                    reuse=True)

    x_cycle_logits_final = x_cycle_logits
    x_cycle_out_final = tf.tanh(x_cycle_logits_final)

    tf.summary.image('x_cycle_out_final',x_cycle_out_final)

    ################### x_hat -> y_cycle  G_A###################


    y_cycle_logits = Generator(x_hat_out_final,
                               keep_prob,
                               layer=layer,
                               scope='G_A',
                               fuse_channals=fuse_channals,
                               output_channel=1,
                               growth_rate=growth_rate,
                               compress_rate=compress_rate,
                               is_training=is_training,
                               reuse=True,
                               dilated_rates= [1,1,1,1,1]
                               )

    y_cycle_logits_final = y_cycle_logits[0]
    y_cycle_out_final = tf.tanh(y_cycle_logits_final)
    tf.summary.image('y_cycle_out_final',y_cycle_out_final)


    ################### D_A ###################

    print('D_A architecature1')

    y_hat_labeled_fea,y_hat_labeled_logits,y_hat_labeled_prob= Discriminator(y_out_labeled_final,
                                                                             base_n_channel=16,
                                                                             keep_prob=keep_prob,
                                                                             scope='D_A',
                                                                             is_training=is_training,
                                                                             reuse=False)

    print('D_B architecature2')

    y_hat_unlabeled_fea, y_hat_unlabeled_logits, y_hat_unlabeled_prob = Discriminator(y_out_unlabeled_final,
                                                                                    base_n_channel=16,
                                                                                    keep_prob=keep_prob,
                                                                                    scope='D_A',
                                                                                    is_training=is_training,
                                                                                    reuse=True)

    ################### D_B ###################

    x_hat_fake_fea,x_hat_fake_logits,x_hat_fake_prob = Discriminator(x_hat_out_final,
                                                                     base_n_channel=16,
                                                                     keep_prob=keep_prob,
                                                                     scope='D_B',
                                                                     is_training=is_training,
                                                                     reuse=False)

    print('D_B architecature2')

    x_real_fea, x_real_logits, x_real_prob = Discriminator(x_l,
                                                            base_n_channel=16,
                                                            keep_prob=keep_prob,
                                                            scope='D_B',
                                                            is_training=is_training,
                                                            reuse=True)


    ###############################
    # define metrics
    ###############################

    y_hat_labeled_MAE = MAE(y,y_hat_labeled_final)
    y_hat_labeled_pre,y_hat_labeled_rec,y_hat_labeled_F = F_measure(y,y_hat_labeled_final)


    ###############################
    # define loss
    ###############################





    ########## perception_loss ##########

    x_and_x_hat = tf.concat([x, x_cycle_out_final, x_hat_out_final],axis=0)
    x_and_x_hat = (x_and_x_hat + 1.0) / 2.0 * 255.0

    with slim.arg_scope(pl.vgg_arg_scope()):

        x_and_x_hat_processed = pl.img_process(x_and_x_hat,True)

        f1,f2,f3,f4,exclude = pl.vgg_16(x_and_x_hat_processed)

        vgg_var = slim.get_variables_to_restore(include=['vgg_16'],exclude=exclude)

        vgg_init_fn = slim.assign_from_checkpoint_fn('vgg_16.ckpt',vgg_var)


    perception_loss1,perception_loss2 = pl.styleloss(f1,f2,f3,f4)
    tf.summary.scalar('perception_loss1', perception_loss1)
    tf.summary.scalar('perception_loss2', perception_loss2)


    # cycle_loss = tf.reduce_mean(tf.square(y_cycle_out_final - y_norm)) + \
    #              tf.reduce_mean(tf.square(x - x_cycle_out_final))

    cycle_loss = tf.reduce_mean(tf.square(y_cycle_out_final - y_norm) + tf.abs(y_cycle_out_final - y_norm)) + \
                 5 * tf.reduce_mean(tf.square(x - x_cycle_out_final) + tf.abs(x - x_cycle_out_final))

    # cycle_loss = tf.reduce_mean(fuse_loss_for_saliency_detection(yp=(y_cycle_out_final + 1) / 2.0,
    #                                                              yp_logits=y_cycle_logits_final,
    #                                                              gt=y)) + \
    #              1.0 * (tf.reduce_mean(tf.abs(x - x_cycle_out_final)) + w3 * gl(x, x_cycle_out_final, channel=3) + w4 * perception_loss1)

    tf.summary.scalar('cycle_loss', cycle_loss)

    ########## loss for D_A ##########

    L_D_A_labeled =  tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_labeled_logits,
                                                                          labels=tf.zeros_like(y_hat_labeled_logits,
                                                                                               dtype=tf.float32)))
    L_D_A_unlabeled = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_unlabeled_logits,
                                                                         labels=tf.ones_like(y_hat_unlabeled_logits,
                                                                                             dtype=tf.float32)))

    L_D_A = L_D_A_labeled + L_D_A_unlabeled

    tf.summary.scalar('L_D_A',L_D_A)


    ########## loss for G_A ##########

    L_G_A_sup = 1.0 * fuse_loss_for_saliency_detection(yp=y_hat_labeled[0, :, :, :, :],
                                                     yp_logits=y_logits_labeled[0, :, :, :, :],gt=y) + \
              0.8 * fuse_loss_for_saliency_detection(yp=y_hat_labeled[1, :, :, :, :],
                                                     yp_logits=y_logits_labeled[1, :, :, :, :], gt=y) + \
              0.6 * fuse_loss_for_saliency_detection(yp=y_hat_labeled[2, :, :, :, :],
                                                     yp_logits=y_logits_labeled[2, :, :, :, :], gt=y) + \
              0.4 * fuse_loss_for_saliency_detection(yp=y_hat_labeled[3, :, :, :, :],
                                                     yp_logits=y_logits_labeled[3, :, :, :, :], gt=y) + \
              0.2 * fuse_loss_for_saliency_detection(yp=y_hat_labeled[4, :, :, :, :],
                                                     yp_logits=y_logits_labeled[4, :, :, :, :], gt=y) + \
              0.1 * fuse_loss_for_saliency_detection(yp=y_hat_labeled[5, :, :, :, :],
                                                     yp_logits=y_logits_labeled[5, :, :, :, :], gt=y)

    tf.summary.scalar('L_G_A_sup', L_G_A_sup)

    L_G_A_adv = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_unlabeled_logits,
                                                                         labels=tf.zeros_like(y_hat_unlabeled_logits,
                                                                                              dtype=tf.float32)))

    tf.summary.scalar('L_G_A_adv', L_G_A_adv)

    L_G_A = L_G_A_sup + w1*L_G_A_adv + w2 * cycle_loss
    #L_G_A = L_G_A_sup + w1*L_G_A_adv
    #L_G_A = L_G_A_sup

    ########## loss for D_B ##########


    L_D_B_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_real_logits,
                                                                           labels=tf.zeros_like(x_real_logits,
                                                                                                dtype=tf.float32)))
    L_D_B_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat_fake_logits,
                                                                             labels=tf.ones_like(x_hat_fake_logits,
                                                                                                 dtype=tf.float32)))

    L_D_B = L_D_B_real + L_D_B_fake
    tf.summary.scalar('L_D_B', L_D_B)



    ########## loss for G_B ##########


    x_MAE = MAE(x_hat_out_final,x_l)
    tf.summary.scalar('x_MAE', x_MAE)

    L_G_B_sup = tf.reduce_mean(  tf.abs( x_hat_out_final - x_l )) + w3*gl(x_hat_out_final,x_l,channel=3)
    tf.summary.scalar('L_G_B_sup', L_G_B_sup)

    L_G_B_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat_fake_logits,
                                                                             labels=tf.zeros_like(x_hat_fake_logits,
                                                                                                 dtype=tf.float32)))
    tf.summary.scalar('L_G_B_adv', L_G_B_adv)


    #L_G_B = L_G_B_sup + w1*L_G_B_adv + w2*cycle_loss + w4 * perception_loss2
    L_G_B = L_G_B_sup + 0.0001*L_G_B_adv + w2*cycle_loss
    #L_G_B = L_G_B_sup + w1*L_G_B_adv + w4 * perception_loss2
    #L_G_B = L_G_B_sup



    ###############################
    # define optimizer
    ###############################
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        all_vars = tf.trainable_variables()
        G_A_vars = [var for var in all_vars if var.name.startswith('G_A')]
        D_A_vars = [var for var in all_vars if var.name.startswith('D_A')]

        G_B_vars = [var for var in all_vars if var.name.startswith('G_B')]
        D_B_vars = [var for var in all_vars if var.name.startswith('D_B')]

        print( 'G_A_vars',G_A_vars)
        G_A_Conv_Vars = [var for var in G_A_vars if 'conv' in var.name]
        print('G_A_Conv_Vars', G_A_Conv_Vars)

        l2 = tf.add_n( [tf.nn.l2_loss(var) for var in G_A_Conv_Vars])


        G_A_optimazer = tf.train.AdamOptimizer(learning_rate=lr,
                                               beta1=beta1,
                                               beta2=beta2).minimize(L_G_A, var_list=G_A_vars)

        D_A_optimazer = tf.train.AdamOptimizer(learning_rate=lr,
                                               beta1=beta1,
                                               beta2=beta2).minimize(L_D_A, var_list=D_A_vars)

        G_B_optimazer = tf.train.AdamOptimizer(learning_rate=lr,
                                               beta1=beta1,
                                               beta2=beta2).minimize(L_G_B, var_list=G_B_vars)

        D_B_optimazer = tf.train.AdamOptimizer(learning_rate=lr,
                                               beta1=beta1,
                                               beta2=beta2).minimize(L_D_B, var_list=D_B_vars)

        with tf.control_dependencies([G_A_optimazer]):
            l2_loss = weight_decay * l2
            sgd = tf.train.GradientDescentOptimizer(learning_rate=lr)
            decay_op = sgd.minimize(l2_loss)






################################
# begin_training
################################


other_images = np.load(data_root +'/other_images.npy',mmap_mode='r')
other_shapes = np.load(data_root +'/other_shapes.npy',mmap_mode='r')

test_images = np.load(data_root + '/test_images.npy',mmap_mode='r')
test_masks = np.load(data_root + '/test_masks.npy',mmap_mode='r')
test_shapes = np.load(data_root + '/test_shapes.npy',mmap_mode='r')

train_images = np.load(data_root + '/train_images.npy',mmap_mode='r')
train_masks = np.load(data_root + '/train_masks.npy',mmap_mode='r')
train_shapes = np.load(data_root + '/train_shapes.npy',mmap_mode='r')



with tf.Session(graph=graph) as ss:

    all_vars = tf.global_variables()
    init_var = [ v for v in all_vars if 'vgg_16' not in v.name]
    init = tf.variables_initializer(var_list=init_var)
    ss.run(init)
    #ss.run(tf.global_variables_initializer())

    vgg_init_fn(ss)

    saver = tf.train.Saver(max_to_keep=300)
    #saver.restore(sess=ss, save_path='sc_v40/model_60.ckpt')

    merge_summary_op = tf.summary.merge_all()
    train_summmary_writer = tf.summary.FileWriter(log_dir + '/train')
    val_summmary_writer = tf.summary.FileWriter(log_dir + '/val')

    train_info = []
    test_info = []

    train_index = 0
    test_index = 0

    for e in range(epcho):

        train_list = []
        for i in range(one_epoche_iterations):
            batch_train_imgs,batch_train_masks,batch_train_other_imgs,batch_codes = load_batch_trainning_samples(train_images,
                                                                                                     train_masks,
                                                                                                     train_shapes,
                                                                                                     other_images,
                                                                                                     other_shapes,
                                                                                                     fix_size=fix_size,
                                                                                                     batch_size=batch_size,
                                                                                                     latent_varibale_length = latent_varibale_length)

            # print('batch_train_imgs',np.shape(batch_train_imgs))
            # print('batch_train_masks',np.shape(batch_train_masks))
            # print('batch_train_other_imgs',np.shape(batch_train_other_imgs))

            feed_dict = { x_l:batch_train_imgs,
                          x_u:batch_train_other_imgs,
                          y: batch_train_masks,
                          c:batch_codes,
                          lr:learning_rate,
                          keep_prob:kp,
                          is_training:True}

            D_A_optimazer.run(feed_dict)
            D_B_optimazer.run(feed_dict)

            G_A_optimazer.run(feed_dict)
            G_B_optimazer.run(feed_dict)
            decay_op.run(feed_dict)

            summary,da,db,ga_sup,gb_sup,mae,pre,rec,f_m = ss.run([merge_summary_op,
                                                                 L_D_A,
                                                                 L_D_B,
                                                                 L_G_A_sup,
                                                                 x_MAE,
                                                                 y_hat_labeled_MAE,
                                                                 y_hat_labeled_pre,
                                                                 y_hat_labeled_rec,
                                                                 y_hat_labeled_F],feed_dict=feed_dict)

            train_summmary_writer.add_summary(summary,global_step= train_index)

            train_index = train_index + 1

            rt_list = [da,db,ga_sup,gb_sup,mae,pre,rec,f_m]

            print(e,i,rt_list)

            train_list.append(rt_list)

        train_list_array = np.array(train_list)
        train_list_mean = np.mean(train_list_array,axis=0)
        train_info.append(train_list_mean)

        saver.save(sess=ss,save_path= version + '/model_' +str(e) + '.ckpt')

        test_list = []
        for sample in load_one_test_samples(test_images,
                                            test_masks,
                                            test_shapes,
                                            fix_size=fix_size,
                                            latent_varibale_length=latent_varibale_length):

            one_test_img,one_test_mask,one_test_other_img,one_test_code = sample
            test_feed_dict = {x_l: one_test_img,
                             x_u: one_test_other_img,
                             y: one_test_mask,
                             c: one_test_code,
                             lr: learning_rate,
                             keep_prob: 1.0,
                             is_training: False}

            test_summary, mae, pre, rec, f_m = ss.run([merge_summary_op,
                                                      y_hat_labeled_MAE,
                                                      y_hat_labeled_pre,
                                                      y_hat_labeled_rec,
                                                      y_hat_labeled_F], feed_dict=test_feed_dict)

            test_rt =[mae,pre,rec,f_m]
            test_list.append(test_rt)

            test_index = test_index +1
            val_summmary_writer.add_summary(test_summary,global_step=test_index)

        test_list_array = np.array(test_list)
        test_list_mean = np.mean(test_list_array,axis=0)
        test_info.append(test_list_mean)

        print(e,train_list_mean,test_list_mean)

        np.save(version + '/train_info',train_info)
        np.save(version + '/test_info',test_info)

