# -*- coding=utf-8 -*-
from util.image_preprocessing_util import *
from util.network_util import *
import  tensorflow as tf
import numpy as np






def Generator_with_Four_Levels(x,keep_prob,layer,scope,fuse_channals,output_channel,growth_rate,compress_rate,is_training=True, reuse=True):

    with tf.variable_scope(scope, reuse=reuse):


        _,_,_,ch = x.get_shape().as_list()

        x = batch_activ_conv(x,
                             n_in_features=ch,
                             n_out_features=16,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             name='l1'
                             ) #256

        x = batch_activ_conv(x,
                             n_in_features=16,
                             n_out_features=32,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             name='l2'
                             )  # s

        ########################
        # block1
        ########################
        x,n_channels = dense_block(x,
                                   layer // 2,
                                   32,
                                   growth_rate,
                                   is_training,
                                   keep_prob,
                                   name = 'db1',
                                   dilated_rate=1)



        n_channels_pressed = np.int32(n_channels * compress_rate)

        x_compressed = batch_activ_conv( x,
                                         n_in_features= n_channels,
                                         n_out_features=n_channels_pressed,
                                         keep_prob=keep_prob,
                                         kernel_size=3,
                                         is_training=is_training,
                                         name='cp1'
                                         )  # s



        x = avg_pool(x_compressed,2,padding='SAME')

        ########################
        # block2
        ########################

        x, n_channels = dense_block(x,
                                    layer // 2,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db2',
                                    dilated_rate=2) # s // 2

        scale2_fea = batch_activ_conv(x,
                                     n_in_features=n_channels,
                                     n_out_features=fuse_channals,
                                     keep_prob=keep_prob,
                                     kernel_size=3,
                                     is_training=is_training,
                                     name='ef2'
                                     )  # s //2

        n_channels_pressed = np.int32(n_channels * compress_rate)
        x_compressed = batch_activ_conv(x,
                                        n_in_features=n_channels,
                                        n_out_features=n_channels_pressed,
                                        keep_prob=keep_prob,
                                        kernel_size=3,
                                        is_training=is_training,
                                        name='cp2'
                                        )  # s // 2

        x = avg_pool(x_compressed, 2, padding='SAME')

        ########################
        # block3
        ########################

        x, n_channels = dense_block(x,
                                    layer,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db3',
                                    dilated_rate=4)  #  s// 4

        scale3_fea = batch_activ_conv(x,
                                      n_in_features=n_channels,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='ef3'
                                      )  # s // 4

        n_channels_pressed = np.int32(n_channels * compress_rate)
        x_compressed = batch_activ_conv(x,
                                        n_in_features=n_channels,
                                        n_out_features=n_channels_pressed,
                                        keep_prob=keep_prob,
                                        kernel_size=3,
                                        is_training=is_training,
                                        name='cp3'
                                        )  # s // 4


        ########################
        # block4
        ########################

        x, n_channels = dense_block(x_compressed,
                                    layer,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db4',
                                    dilated_rate=8)  # s // 4

        scale4_fea = batch_activ_conv(x,
                                      n_in_features=n_channels,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='ef4'
                                      )  # s // 4

        n_channels_pressed = np.int32(n_channels * compress_rate)
        x_compressed = batch_activ_conv(x,
                                        n_in_features=n_channels,
                                        n_out_features=n_channels_pressed,
                                        keep_prob=keep_prob,
                                        kernel_size=3,
                                        is_training=is_training,
                                        name='cp4'
                                        )  # s // 4

        ########################
        # block5
        ########################

        x, n_channels = dense_block(x_compressed,
                                    layer,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db5',
                                    dilated_rate=16)  # s // 4

        scale5_fea = batch_activ_conv(x,
                                      n_in_features=n_channels,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='ef5'
                                      ) # s // 4


        ########################
        # ppm
        ########################
        _, ht, wd, ch = scale5_fea.get_shape().as_list()

        assert ht == wd

        if ht == 64:
            ppm_fea,n_channels_ppm = pyramid_pooling_64(scale5_fea,fuse_channals,name='ppm64')
        elif ht == 32:
            ppm_fea, n_channels_ppm = pyramid_pooling_32(scale5_fea, fuse_channals, name='ppm32')
        else:
            print('Wrong feature shape!')
            exit(-1)

        ########################
        # fuse_stage1
        ########################


        scale5_fea_fused = tf.concat([scale5_fea,ppm_fea],axis=3)
        n_channels_scale5_fea_fused = fuse_channals + n_channels_ppm

        scale5_map_logits  = batch_activ_conv(scale5_fea_fused,
                                      n_in_features=n_channels_scale5_fea_fused,
                                      n_out_features= output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu1',
                                      with_bias=True
                                      ) #s // 4

        scale5_fea_fused = batch_activ_conv(scale5_fea_fused,
                                      n_in_features=n_channels_scale5_fea_fused,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='tf1'
                                      )  #s // 4


        ########################
        # fuse_stage2
        ########################

        scale4_fea_fused = tf.concat([scale4_fea, scale5_fea_fused], axis=3)
        n_channels_scale4_fea_fused = fuse_channals + fuse_channals

        scale4_map_logits  = batch_activ_conv(scale4_fea_fused,
                                      n_in_features=n_channels_scale4_fea_fused,
                                      n_out_features=output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu2',
                                      with_bias=True
                                      )  # s // 4

        scale4_fea_fused = batch_activ_conv(scale4_fea_fused,
                                            n_in_features=n_channels_scale4_fea_fused,
                                            n_out_features=fuse_channals,
                                            keep_prob=keep_prob,
                                            kernel_size=3,
                                            is_training=is_training,
                                            name='tf2'
                                            )# s // 4

        ########################
        # fuse_stage3
        ########################

        scale3_fea_fused = tf.concat([scale3_fea, scale4_fea_fused], axis=3)
        n_channels_scale3_fea_fused = fuse_channals + fuse_channals

        scale3_map_logits  = batch_activ_conv(scale3_fea_fused,
                                      n_in_features=n_channels_scale3_fea_fused,
                                      n_out_features=output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu3',
                                      with_bias=True
                                      )  # s // 4

        scale3_fea_fused = batch_activ_conv(scale3_fea_fused,
                                            n_in_features=n_channels_scale3_fea_fused,
                                            n_out_features=fuse_channals,
                                            keep_prob=keep_prob,
                                            kernel_size=3,
                                            is_training=is_training,
                                            name='tf3'
                                            )  # s // 4

        ########################
        # fuse_stage4
        ########################

        scale3_fea_fused_upsampled = upsample(scale3_fea_fused,
                                              factor=2,
                                              channel=fuse_channals)

        scale2_fea_fused = tf.concat([scale2_fea, scale3_fea_fused_upsampled], axis=3)
        n_channels_scale2_fea_fused = fuse_channals + fuse_channals

        scale2_map_logits = batch_activ_conv(scale2_fea_fused,
                                      n_in_features=n_channels_scale2_fea_fused,
                                      n_out_features=output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu4',
                                      with_bias=True
                                      )


        scale2_map_logits_upsampled = upsample(scale2_map_logits,factor=2,channel=output_channel)
        scale3_map_logits_upsampled = upsample(scale3_map_logits,factor=4,channel=output_channel)
        scale4_map_logits_upsampled = upsample(scale4_map_logits,factor=4,channel=output_channel)
        scale5_map_logits_upsampled = upsample(scale5_map_logits,factor=4,channel=output_channel)


        all_scale_logits = tf.concat([
                                      scale2_map_logits_upsampled,
                                      scale3_map_logits_upsampled,
                                      scale4_map_logits_upsampled,
                                      scale5_map_logits_upsampled
                                      ],
                                     axis=3
                                     )


        final_map_logits = conv2d(all_scale_logits,
                                  4 * output_channel,
                                  output_channel,
                                  kernel_size=3,
                                  name='final_logits',
                                  with_bias=True)


        all_logits = [final_map_logits,
                      scale2_map_logits_upsampled,
                      scale3_map_logits_upsampled,
                      scale4_map_logits_upsampled,
                      scale5_map_logits_upsampled
                      ]

        all_logits = [ tf.expand_dims(e,axis=0) for e in all_logits]
        all_logits = tf.concat(all_logits,axis=0)

        print('all_logits',all_logits)

        return all_logits


def Generator(x,
              keep_prob,
              layer,
              scope,
              fuse_channals,
              output_channel,
              growth_rate,
              compress_rate,
              is_training=True,
              reuse=True,
              dilated_rates = [1,2,4,8,16]):

    with tf.variable_scope(scope, reuse=reuse):


        _,_,_,ch = x.get_shape().as_list()

        x = batch_activ_conv(x,
                             n_in_features=ch,
                             n_out_features=16,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             name='l1'
                             ) #256

        x = batch_activ_conv(x,
                             n_in_features=16,
                             n_out_features=32,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             name='l2'
                             )  # s

        ########################
        # block1
        ########################
        x,n_channels = dense_block(x,
                                   layer // 2,
                                   32,
                                   growth_rate,
                                   is_training,
                                   keep_prob,
                                   name = 'db1',
                                   dilated_rate=dilated_rates[0])

        scale1_fea = batch_activ_conv(x,
                                     n_in_features=n_channels,
                                     n_out_features=fuse_channals,
                                     keep_prob=keep_prob,
                                     kernel_size=3,
                                     is_training=is_training,
                                     name='ef1'
                                     )  # s

        n_channels_pressed = np.int32(n_channels * compress_rate)

        x_compressed = batch_activ_conv( x,
                                         n_in_features= n_channels,
                                         n_out_features=n_channels_pressed,
                                         keep_prob=keep_prob,
                                         kernel_size=3,
                                         is_training=is_training,
                                         name='cp1'
                                         )  # s



        x = avg_pool(x_compressed,2,padding='SAME')

        ########################
        # block2
        ########################

        x, n_channels = dense_block(x,
                                    layer // 2,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db2',
                                    dilated_rate=dilated_rates[1]) # s // 2

        scale2_fea = batch_activ_conv(x,
                                     n_in_features=n_channels,
                                     n_out_features=fuse_channals,
                                     keep_prob=keep_prob,
                                     kernel_size=3,
                                     is_training=is_training,
                                     name='ef2'
                                     )  # s //2

        n_channels_pressed = np.int32(n_channels * compress_rate)
        x_compressed = batch_activ_conv(x,
                                        n_in_features=n_channels,
                                        n_out_features=n_channels_pressed,
                                        keep_prob=keep_prob,
                                        kernel_size=3,
                                        is_training=is_training,
                                        name='cp2'
                                        )  # s // 2

        x = avg_pool(x_compressed, 2, padding='SAME')

        ########################
        # block3
        ########################

        x, n_channels = dense_block(x,
                                    layer,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db3',
                                    dilated_rate=dilated_rates[2])  #  s// 4

        scale3_fea = batch_activ_conv(x,
                                      n_in_features=n_channels,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='ef3'
                                      )  # s // 4

        n_channels_pressed = np.int32(n_channels * compress_rate)
        x_compressed = batch_activ_conv(x,
                                        n_in_features=n_channels,
                                        n_out_features=n_channels_pressed,
                                        keep_prob=keep_prob,
                                        kernel_size=3,
                                        is_training=is_training,
                                        name='cp3'
                                        )  # s // 4


        ########################
        # block4
        ########################

        x, n_channels = dense_block(x_compressed,
                                    layer,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db4',
                                    dilated_rate=dilated_rates[3])  # s // 4

        scale4_fea = batch_activ_conv(x,
                                      n_in_features=n_channels,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='ef4'
                                      )  # s // 4

        n_channels_pressed = np.int32(n_channels * compress_rate)
        x_compressed = batch_activ_conv(x,
                                        n_in_features=n_channels,
                                        n_out_features=n_channels_pressed,
                                        keep_prob=keep_prob,
                                        kernel_size=3,
                                        is_training=is_training,
                                        name='cp4'
                                        )  # s // 4

        ########################
        # block5
        ########################

        x, n_channels = dense_block(x_compressed,
                                    layer,
                                    n_channels_pressed,
                                    growth_rate,
                                    is_training,
                                    keep_prob,
                                    name='db5',
                                    dilated_rate=dilated_rates[4])  # s // 4

        scale5_fea = batch_activ_conv(x,
                                      n_in_features=n_channels,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='ef5'
                                      ) # s // 4


        ########################
        # ppm
        ########################
        _, ht, wd, ch = scale5_fea.get_shape().as_list()

        assert ht == wd

        if ht == 64:
            ppm_fea,n_channels_ppm = pyramid_pooling_64(scale5_fea,fuse_channals,name='ppm64')
        elif ht == 32:
            ppm_fea, n_channels_ppm = pyramid_pooling_32(scale5_fea, fuse_channals, name='ppm32')
        else:
            print('Wrong feature shape!')
            exit(-1)

        ########################
        # fuse_stage1
        ########################


        scale5_fea_fused = tf.concat([scale5_fea,ppm_fea],axis=3)
        n_channels_scale5_fea_fused = fuse_channals + n_channels_ppm

        scale5_map_logits  = batch_activ_conv(scale5_fea_fused,
                                      n_in_features=n_channels_scale5_fea_fused,
                                      n_out_features= output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu1',
                                      with_bias=True
                                      ) #s // 4

        scale5_fea_fused = batch_activ_conv(scale5_fea_fused,
                                      n_in_features=n_channels_scale5_fea_fused,
                                      n_out_features=fuse_channals,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='tf1'
                                      )  #s // 4


        ########################
        # fuse_stage2
        ########################

        scale4_fea_fused = tf.concat([scale4_fea, scale5_fea_fused], axis=3)
        n_channels_scale4_fea_fused = fuse_channals + fuse_channals

        scale4_map_logits  = batch_activ_conv(scale4_fea_fused,
                                      n_in_features=n_channels_scale4_fea_fused,
                                      n_out_features=output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu2',
                                      with_bias=True
                                      )  # s // 4

        scale4_fea_fused = batch_activ_conv(scale4_fea_fused,
                                            n_in_features=n_channels_scale4_fea_fused,
                                            n_out_features=fuse_channals,
                                            keep_prob=keep_prob,
                                            kernel_size=3,
                                            is_training=is_training,
                                            name='tf2'
                                            )# s // 4

        ########################
        # fuse_stage3
        ########################

        scale3_fea_fused = tf.concat([scale3_fea, scale4_fea_fused], axis=3)
        n_channels_scale3_fea_fused = fuse_channals + fuse_channals

        scale3_map_logits  = batch_activ_conv(scale3_fea_fused,
                                      n_in_features=n_channels_scale3_fea_fused,
                                      n_out_features=output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu3',
                                      with_bias=True
                                      )  # s // 4

        scale3_fea_fused = batch_activ_conv(scale3_fea_fused,
                                            n_in_features=n_channels_scale3_fea_fused,
                                            n_out_features=fuse_channals,
                                            keep_prob=keep_prob,
                                            kernel_size=3,
                                            is_training=is_training,
                                            name='tf3'
                                            )  # s // 4

        ########################
        # fuse_stage4
        ########################

        scale3_fea_fused_upsampled = upsample(scale3_fea_fused,
                                              factor=2,
                                              channel=fuse_channals)

        scale2_fea_fused = tf.concat([scale2_fea, scale3_fea_fused_upsampled], axis=3)
        n_channels_scale2_fea_fused = fuse_channals + fuse_channals

        scale2_map_logits = batch_activ_conv(scale2_fea_fused,
                                      n_in_features=n_channels_scale2_fea_fused,
                                      n_out_features=output_channel,
                                      keep_prob=keep_prob,
                                      kernel_size=3,
                                      is_training=is_training,
                                      name='fu4',
                                      with_bias=True
                                      )

        scale2_fea_fused = batch_activ_conv(scale2_fea_fused,
                                            n_in_features=n_channels_scale2_fea_fused,
                                            n_out_features=fuse_channals,
                                            keep_prob=keep_prob,
                                            kernel_size=3,
                                            is_training=is_training,
                                            name='tf4'
                                            ) #s // 2

        ########################
        # fuse_stage5
        ########################

        scale2_fea_fused_upsampled = upsample(scale2_fea_fused,
                                              factor=2,
                                              channel=fuse_channals)

        scale1_fea_fused = tf.concat([scale1_fea, scale2_fea_fused_upsampled], axis=3)
        n_channels_scale1_fea_fused = fuse_channals + fuse_channals

        scale1_map_logits = batch_activ_conv(scale1_fea_fused,
                                             n_in_features=n_channels_scale1_fea_fused,
                                             n_out_features=output_channel,
                                             keep_prob=keep_prob,
                                             kernel_size=3,
                                             is_training=is_training,
                                             name='fu5',
                                             with_bias=True
                                             )  # s


        scale2_map_logits_upsampled = upsample(scale2_map_logits,factor=2,channel=output_channel)
        scale3_map_logits_upsampled = upsample(scale3_map_logits,factor=4,channel=output_channel)
        scale4_map_logits_upsampled = upsample(scale4_map_logits,factor=4,channel=output_channel)
        scale5_map_logits_upsampled = upsample(scale5_map_logits,factor=4,channel=output_channel)


        all_scale_logits = tf.concat([scale1_map_logits,
                                      scale2_map_logits_upsampled,
                                      scale3_map_logits_upsampled,
                                      scale4_map_logits_upsampled,
                                      scale5_map_logits_upsampled
                                      ],
                                     axis=3
                                     )


        final_map_logits = conv2d(all_scale_logits,
                                  5 * output_channel,
                                  output_channel,
                                  kernel_size=3,
                                  name='final_logits',
                                  with_bias=True)


        all_logits = [final_map_logits,
                      scale1_map_logits,
                      scale2_map_logits_upsampled,
                      scale3_map_logits_upsampled,
                      scale4_map_logits_upsampled,
                      scale5_map_logits_upsampled
                      ]

        all_logits = [ tf.expand_dims(e,axis=0) for e in all_logits]
        all_logits = tf.concat(all_logits,axis=0)

        return all_logits


##################################
# code to image
##################################

#
# def Generator_DC_128(x,c,base_n_channel,output_channel,keep_prob,scope,is_training=True, reuse=True):
#
#     with tf.variable_scope(scope, reuse=reuse):
#         _, dim = c.get_shape().as_list()
#
#         c = tf.reshape(c,shape=[-1,1,1,dim])
#
#         scale1 = conv2d_trans(c,
#                               n_in_features=dim,
#                               n_out_features=base_n_channel * 8,
#                               kernel_size=4,
#                               name='l1',
#                               stride=4) #4x4
#
#         scale2 = batch_activ_conv_trans(scale1,
#                                         n_in_features= base_n_channel * 8,
#                                         n_out_features=base_n_channel * 4,
#                                         kernel_size=4,
#                                         is_training=is_training,
#                                         keep_prob=keep_prob,
#                                         name='l2',
#                                         stride=2
#                                         ) #8x8
#
#         scale3 = batch_activ_conv_trans(scale2,
#                                         n_in_features= base_n_channel * 4,
#                                         n_out_features=base_n_channel * 4,
#                                         kernel_size=4,
#                                         is_training=is_training,
#                                         keep_prob=keep_prob,
#                                         name='l3',
#                                         stride=2
#                                         )  # 16x16
#
#         scale4 = batch_activ_conv_trans(scale3,
#                                         n_in_features= base_n_channel * 4,
#                                         n_out_features=base_n_channel * 2,
#                                         kernel_size=4,
#                                         is_training=is_training,
#                                         keep_prob=keep_prob,
#                                         name='l4',
#                                         stride=2
#                                         )  # 32x32
#
#         scale5 = batch_activ_conv_trans(scale4,
#                                         n_in_features =base_n_channel * 2,
#                                         n_out_features=base_n_channel * 2,
#                                         kernel_size=4,
#                                         is_training=is_training,
#                                         keep_prob=keep_prob,
#                                         name='l5',
#                                         stride=2
#                                         )  # 64x64
#
#         scale6 = batch_activ_conv_trans(scale5,
#                                         n_in_features= base_n_channel * 2,
#                                         n_out_features=base_n_channel,
#                                         kernel_size=4,
#                                         is_training=is_training,
#                                         keep_prob=keep_prob,
#                                         name='l6',
#                                         stride=2
#                                         )  # 128x128
#
#
#         map = conv2d(scale6,
#                      n_in_features= base_n_channel,
#                      n_out_features=output_channel,
#                      kernel_size=3,
#                      name='l7',
#                      with_bias=True)
#
#         return map




##################################
# code and map to image
##################################



def Generator_DC_with_code_128(x,c,base_n_channel,output_channel,keep_prob,scope,is_training=True, reuse=True):

    with tf.variable_scope(scope, reuse=reuse):

        _, _,_,ch = x.get_shape().as_list()
        _, dim = c.get_shape().as_list()

        c = tf.reshape(c,shape=[-1,1,1,dim])

        scale1 = conv2d_trans(c,
                              n_in_features=dim,
                              n_out_features=base_n_channel * 8,
                              kernel_size=4,
                              name='l1',
                              stride=4) #4x4

        scale1_map = batch_activ_conv(x,
                                     n_in_features=ch,
                                     n_out_features=base_n_channel,
                                     kernel_size=3,
                                     is_training=is_training,
                                     keep_prob=keep_prob,
                                     stride=32,
                                     name='m_l1')

        scale1_fused = tf.concat([scale1,scale1_map],axis=3)


        scale2 = batch_activ_conv_trans(scale1_fused,
                                        n_in_features= base_n_channel * 8 + base_n_channel,
                                        n_out_features=base_n_channel * 4,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l2',
                                        stride=2
                                        ) #8x8

        scale2_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=16,
                                      name='m_l2')

        scale2_fused = tf.concat([scale2, scale2_map], axis=3)


        scale3 = batch_activ_conv_trans(scale2_fused,
                                        n_in_features= base_n_channel * 4 + base_n_channel,
                                        n_out_features=base_n_channel * 4,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l3',
                                        stride=2
                                        )  # 16x16

        scale3_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 2,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=8,
                                      name='m_l3')

        scale3_fused = tf.concat([scale3, scale3_map], axis=3)


        scale4 = batch_activ_conv_trans(scale3_fused,
                                        n_in_features= base_n_channel * 4 + base_n_channel // 2,
                                        n_out_features=base_n_channel * 2,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l4',
                                        stride=2
                                        )  # 32x32

        scale4_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 2,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=4,
                                      name='m_l4')

        scale4_fused = tf.concat([scale4, scale4_map], axis=3)


        scale5 = batch_activ_conv_trans(scale4_fused,
                                        n_in_features =base_n_channel * 2 + base_n_channel // 2,
                                        n_out_features=base_n_channel * 2,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l5',
                                        stride=2
                                        )  # 64x64


        scale5_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 4,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=2,
                                      name='m_l5')

        scale5_fused = tf.concat([scale5, scale5_map], axis=3)



        scale6 = batch_activ_conv_trans(scale5_fused,
                                        n_in_features= base_n_channel * 2 + base_n_channel // 4,
                                        n_out_features=base_n_channel,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l6',
                                        stride=2
                                        )  # 128x128

        scale6_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 4,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=1,
                                      name='m_l6')

        scale6_fused = tf.concat([scale6, scale6_map], axis=3)

        map = conv2d(scale6_fused,
                     n_in_features= base_n_channel + base_n_channel // 4,
                     n_out_features=output_channel,
                     kernel_size=3,
                     name='l7',
                     with_bias=True)

        return map



##################################
# map to image
##################################


def Generator_DC_128(x,c,base_n_channel,output_channel,keep_prob,scope,is_training=True, reuse=True):

    with tf.variable_scope(scope, reuse=reuse):

        _, ht,wd,ch = x.get_shape().as_list()

        scale0_map = conv2d(x,
                            n_in_features=ch,
                            n_out_features= base_n_channel*4,
                            kernel_size=3,
                            stride=1,
                            name='m_l0')

        scale0_map = tf.reduce_mean(scale0_map,axis=(1,2))

        c = tf.reshape(scale0_map, shape=[-1, 1, 1, base_n_channel*4])

        scale1 = conv2d_trans(c,
                              n_in_features=base_n_channel*4,
                              n_out_features=base_n_channel * 8,
                              kernel_size=4,
                              name='l1',
                              stride=4) #4x4

        scale1_map = batch_activ_conv(x,
                                     n_in_features=ch,
                                     n_out_features=base_n_channel,
                                     kernel_size=3,
                                     is_training=is_training,
                                     keep_prob=keep_prob,
                                     stride=32,
                                     name='m_l1')

        scale1_fused = tf.concat([scale1,scale1_map],axis=3)

        scale2 = batch_activ_conv_trans(scale1_fused,
                                        n_in_features= base_n_channel * 8 + base_n_channel,
                                        n_out_features=base_n_channel * 4,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l2',
                                        stride=2
                                        ) #8x8

        scale2_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=16,
                                      name='m_l2')

        scale2_fused = tf.concat([scale2, scale2_map], axis=3)


        scale3 = batch_activ_conv_trans(scale2_fused,
                                        n_in_features= base_n_channel * 4 + base_n_channel,
                                        n_out_features=base_n_channel * 4,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l3',
                                        stride=2
                                        )  # 16x16

        scale3_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 2,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=8,
                                      name='m_l3')

        scale3_fused = tf.concat([scale3, scale3_map], axis=3)


        scale4 = batch_activ_conv_trans(scale3_fused,
                                        n_in_features= base_n_channel * 4 + base_n_channel // 2,
                                        n_out_features=base_n_channel * 2,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l4',
                                        stride=2
                                        )  # 32x32

        scale4_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 2,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=4,
                                      name='m_l4')

        scale4_fused = tf.concat([scale4, scale4_map], axis=3)


        scale5 = batch_activ_conv_trans(scale4_fused,
                                        n_in_features =base_n_channel * 2 + base_n_channel // 2,
                                        n_out_features=base_n_channel * 2,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l5',
                                        stride=2
                                        )  # 64x64


        scale5_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 4,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=2,
                                      name='m_l5')

        scale5_fused = tf.concat([scale5, scale5_map], axis=3)



        scale6 = batch_activ_conv_trans(scale5_fused,
                                        n_in_features= base_n_channel * 2 + base_n_channel // 4,
                                        n_out_features=base_n_channel,
                                        kernel_size=4,
                                        is_training=is_training,
                                        keep_prob=keep_prob,
                                        name='l6',
                                        stride=2
                                        )  # 128x128

        scale6_map = batch_activ_conv(x,
                                      n_in_features=ch,
                                      n_out_features=base_n_channel // 4,
                                      kernel_size=3,
                                      is_training=is_training,
                                      keep_prob=keep_prob,
                                      stride=1,
                                      name='m_l6')

        scale6_fused = tf.concat([scale6, scale6_map], axis=3)

        map = conv2d(scale6_fused,
                     n_in_features= base_n_channel + base_n_channel // 4,
                     n_out_features=output_channel,
                     kernel_size=3,
                     name='l7',
                     with_bias=True)

        return map











def Discriminator(x,base_n_channel,keep_prob,scope,is_training=True, reuse=True):

    with tf.variable_scope(scope, reuse=reuse):
        _, _, _, ch = x.get_shape().as_list()
        x = batch_activ_conv(x,
                             n_in_features=ch,
                             n_out_features=base_n_channel,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             stride=2,
                             name='l1'
                             ) #256

        x = batch_activ_conv(x,
                             n_in_features=base_n_channel,
                             n_out_features=2 * base_n_channel,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             stride=2,
                             name='l2'
                             ) # 128

        x = batch_activ_conv(x,
                             n_in_features=2 * base_n_channel,
                             n_out_features=4 * base_n_channel,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             stride=2,
                             name='l3'
                             ) #64

        x = batch_activ_conv(x,
                             n_in_features=4 * base_n_channel,
                             n_out_features=8 * base_n_channel,
                             keep_prob=keep_prob,
                             kernel_size=3,
                             is_training=is_training,
                             stride=4,
                             name='l4'
                             ) # 16

        flatten = tf.reduce_mean(x, axis=[1, 2])

        logits = tf.layers.dense(flatten, 1)

        prob = tf.nn.sigmoid(logits)

        return flatten,logits,prob




