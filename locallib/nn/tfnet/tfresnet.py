import tensorflow as tf
from locallib.nn.tfloss.tfloss import TfLosses

class TfResNet:
    @staticmethod
    def Act(data, act_type, name):
        assert act_type in ['prelu', 'relu', 'sigmoid'], 'only prelu and relu are supported'
        if act_type=='prelu':
            body = tf.keras.layers.PReLU(name=name)(data)
        elif act_type=='relu':
            body = tf.nn.relu(data, name=name)
        elif act_type=='sigmoid':
            body = tf.nn.sigmoid(data, name=name)
        return body

    @staticmethod
    def residual_module_v3(data, K, stride, training, initializer=None, act_type='prelu', red=False, bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False, scope=None):
        
        # the shortcut branch of the ResNet module should be initialized as the input (identity) data
        shortcut = data

        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=data, epsilon=bnEps, momentum=bnMom, training=training, name='bn1')
            if bottle_neck:
                # the first block of the ResNet module are 1x1 CONVs
                conv1 = tf.layers.conv2d(   inputs=bn1, padding='same', kernel_size=(1, 1), strides=(1, 1), filters=int(K * 0.25), 
                                            kernel_initializer=initializer, use_bias=False, name='conv1')

                # the second block of the ResNet module are 3x3 CONVs
                bn2 = tf.layers.batch_normalization(inputs=conv1, epsilon=bnEps, momentum=bnMom, training=training, name='bn2')
                act2 = TfResNet.Act(bn2, act_type, name='%s2' % (act_type))
                conv2 = tf.layers.conv2d(   inputs=act2, padding='same', kernel_size=(3, 3), strides=(1, 1), filters=int(K * 0.25), 
                                            kernel_initializer=initializer, use_bias=False, name='conv2')

                # the third block of the ResNet module is another set of 1x1 CONVs
                bn3 = tf.layers.batch_normalization(inputs=conv2, epsilon=bnEps, momentum=bnMom, training=training, name='bn3')
                act3 = TfResNet.Act(bn3, act_type, name='%s3' % (act_type))
                conv3 = tf.layers.conv2d(   inputs=act3, padding='same', kernel_size=(1, 1), strides=stride, filters=K, 
                                            kernel_initializer=initializer, use_bias=False, name='conv3')

                # the forth block for bn
                bn4 = tf.layers.batch_normalization(inputs=conv3, epsilon=bnEps, momentum=bnMom, training=training, name='bn4')
            else:
                conv1 = tf.layers.conv2d(   inputs=bn1, padding='same', kernel_size=(3, 3), strides=(1, 1), filters=K, \
                                            kernel_initializer=initializer, use_bias=False, name='conv1')

                # the second block of the ResNet module are 3x3 CONVs
                bn2 = tf.layers.batch_normalization(inputs=conv1, epsilon=bnEps, momentum=bnMom, training=training, name='bn2')
                act2 = TfResNet.Act(bn2, act_type, name='%s2' % (act_type))
                conv2 = tf.layers.conv2d(   inputs=act2, padding='same', kernel_size=(3, 3), strides=stride, filters=K, 
                                            kernel_initializer=initializer, use_bias=False, name='conv2')

                # the third block of the ResNet module is another set of 1x1 CONVs
                bn4 = tf.layers.batch_normalization(inputs=conv2, epsilon=bnEps, momentum=bnMom, training=training, name='bn4')

            # add Squeeze-and-Excitation module
            if use_se:
                #se begin
                body = tf.reduce_mean(bn4, axis=[1,2], name='se_pool1')
                body = tf.expand_dims(body, 1)
                body = tf.expand_dims(body, 1)
                body = tf.layers.conv2d(inputs=body, padding='same', kernel_size=(1, 1), strides=(1, 1), filters=K//16, 
                                        kernel_initializer=initializer, name='se_conv1')
                body = TfResNet.Act(body, act_type, name='se_%s1' % (act_type))
                body = tf.layers.conv2d(inputs=body, padding='same', kernel_size=(1, 1), strides=(1, 1), filters=K, 
                                        kernel_initializer=initializer, name='se_conv2')
                body = TfResNet.Act(body, 'sigmoid', name='se_sigmoid1')
                bn4 = tf.multiply(bn4, body, name='se_bc1')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if red:
                shortcut = tf.layers.conv2d(inputs=data, padding='same', kernel_size=(1, 1), strides=stride, filters=K, 
                                            kernel_initializer=initializer, use_bias=False, name='convr')
                shortcut = tf.layers.batch_normalization(inputs=shortcut, epsilon=bnEps, momentum=bnMom, training=training, name='bnr')

            # add together the shortcut and the final CONV
            add = bn4 + shortcut

        # return the addition as the output of the ResNet module
        return add

    @staticmethod
    def residual_module_v2(data, K, stride, training, initializer=None, act_type='prelu', red=False, bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False, scope=None):
        # the shortcut branch of the ResNet module should be initialized as the input (identity) data
        shortcut = data
        
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=data, epsilon=bnEps, momentum=bnMom, training=training, name='bn1')
            act1 = TfResNet.Act(data=bn1, act_type=act_type, name='%s1' % (act_type))
            if bottle_neck:
                # the first block of the ResNet module are 1x1 CONVs
                conv1 = tf.layers.conv2d(   inputs=act1, padding='same', kernel_size=(1, 1), strides=(1, 1), filters=int(K * 0.25), 
                                            kernel_initializer=initializer, use_bias=False, name='conv1')
                # the second block of the ResNet module are 3x3 CONVs
                bn2 = tf.layers.batch_normalization(inputs=conv1, epsilon=bnEps, momentum=bnMom, training=training, name='bn2')
                act2 = TfResNet.Act(data=bn2, act_type=act_type, name='%s2' % (act_type))
                conv2 = tf.layers.conv2d(   inputs=act2, padding='same', kernel_size=(3, 3), strides=stride, filters=int(K * 0.25), 
                                            kernel_initializer=initializer, use_bias=False, name='conv2')

                # the third block of the ResNet module is another set of 1x1 CONVs
                bn3 = tf.layers.batch_normalization(inputs=conv2, epsilon=bnEps, momentum=bnMom, training=training, name='bn3')
                act3 = TfResNet.Act(data=bn3, act_type=act_type, name='%s3' % (act_type))
                conv3 = tf.layers.conv2d(   inputs=act3, padding='same', kernel_size=(1, 1), strides=(1, 1), filters=K, 
                                            kernel_initializer=initializer, use_bias=False, name='conv3')
            else:
                conv1 = tf.layers.conv2d(   inputs=act1, padding='same', kernel_size=(3, 3), strides=stride, filters=K, 
                                            kernel_initializer=initializer, use_bias=False, name='conv1')

                # the second block of the ResNet module are 3x3 CONVs
                bn2 = tf.layers.batch_normalization(inputs=conv1, epsilon=bnEps, momentum=bnMom, training=training, name='bn2')
                act2 = TfResNet.Act(data=bn2, act_type=act_type, name='%s2' % (act_type))
                conv3 = tf.layers.conv2d(   inputs=act2, padding='same', kernel_size=(3, 3), strides=(1, 1), filters=K, 
                                            kernel_initializer=initializer, use_bias=False, name='conv2')

            # add Squeeze-and-Excitation module
            if use_se:
                #se begin
                body = tf.reduce_mean(conv3, axis=[1,2], name='se_pool1')
                body = tf.layers.conv2d(inputs=body, padding='same', kernel_size=(1, 1), strides=(1, 1), filters=K//16, 
                                        kernel_initializer=initializer, name='se_conv1')
                body = TfResNet.Act(data=body, act_type=act_type, name='se_%s1' % (act_type))
                body = tf.layers.conv2d(inputs=body, padding='same', kernel_size=(1, 1), strides=(1, 1), filters=K, 
                                        kernel_initializer=initializer, name='se_conv2')
                body = TfResNet.Act(data=body, act_type='sigmoid', name='se_sigmoid1')
                conv3 = tf.multiply(conv3, body, name='se_bc1')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if red:
                shortcut = tf.layers.conv2d(inputs=act1, padding='same', kernel_size=(1, 1), strides=stride, filters=K, 
                                            kernel_initializer=initializer, use_bias=False, name='convr')

            # add together the shortcut and the final CONV
            add = conv3 + shortcut

            # return the addition as the output of the ResNet module
            return add

    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    @staticmethod
    def build_modules(data, feature_size, stages, filters, training, initializer=None, reuse=None, res_ver='v2', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False, scope=None):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'
        if res_ver=='v2':
            residual_module = TfResNet.residual_module_v2
        elif res_ver=='v3':
            residual_module = TfResNet.residual_module_v3

        with tf.variable_scope(scope, reuse=reuse):            
            # data input
            #data = mx.sym.Variable("data")

            # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
            #bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom, name='stem_bn1')
            if in_ver=='v1':
                conv1_1 = tf.layers.conv2d( inputs=data, padding='same', kernel_size=(7, 7), strides=(2, 2), filters=filters[0], 
                                            kernel_initializer=initializer, use_bias=False, name='stem_conv1')
            elif in_ver=='v2':
                conv1_1 = tf.layers.conv2d( inputs=data, padding='same', kernel_size=(3, 3), strides=(1, 1), filters=filters[0], 
                                            kernel_initializer=initializer, use_bias=False, name='stem_conv1')
            bn1_2 = tf.layers.batch_normalization(inputs=conv1_1, epsilon=bnEps, momentum=bnMom, training=training, name='stem_bn2')
            act1_2 = TfResNet.Act(data=bn1_2, act_type="prelu", name='stem_relu1')
            #pool1 = mx.sym.Pooling(data=act1_2, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))
            body = act1_2

            # loop over the number of stages
            stride = (2, 2)
            for i in range(0, len(stages)):
                # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume            
                body = residual_module( body, filters[i + 1], stride, training, initializer=initializer, red=True, bnEps=bnEps, bnMom=bnMom, 
                                        bottle_neck=bottle_neck, use_se=use_se, scope='stage%d_unit%d' % (i+1, 1))
                # loop over the number of layers in the stage
                for j in range(0, stages[i] - 1):
                    # apply a ResNet module
                    body = residual_module( body, filters[i + 1], (1, 1), training, initializer=initializer, bnEps=bnEps, bnMom=bnMom, 
                                            bottle_neck=bottle_neck, use_se=use_se, scope='stage%d_unit%d' % (i+1, j+2))

            bn2_1 = tf.layers.batch_normalization(inputs=body, epsilon=bnEps, momentum=bnMom, training=training, name='out_bn2')
            act2_1 = TfResNet.Act(data=bn2_1, act_type="prelu", name='out_relu2')
            conv2_1 = tf.layers.conv2d( inputs=act2_1, padding='same', kernel_size=(3, 3), strides=(1, 1), filters=filters[-1], 
                                        kernel_initializer=initializer, use_bias=False, name='out_conv1')
            # apply BN => ACT => POOL
            bn3_1 = tf.layers.batch_normalization(inputs=conv2_1, epsilon=bnEps, momentum=bnMom, training=training, name='out_bn3')
            act3_1 = TfResNet.Act(data=bn3_1, act_type="relu", name='out_relu3')

            # return the network architecture
            return act3_1

    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    @staticmethod
    def build_embedding(data, feature_size, stages, filters, training, initializer=None, reuse=None, res_ver='v2', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False, scope=None):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'
        if res_ver=='v2':
            residual_module = TfResNet.residual_module_v2
        elif res_ver=='v3':
            residual_module = TfResNet.residual_module_v3

        with tf.variable_scope(scope, reuse=reuse):            
            # data input
            #data = mx.sym.Variable("data")

            # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
            #bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom, name='stem_bn1')
            if in_ver=='v1':
                conv1_1 = tf.layers.conv2d( inputs=data, padding='same', kernel_size=(7, 7), strides=(2, 2), filters=filters[0], 
                                            kernel_initializer=initializer, use_bias=False, name='stem_conv1')
            elif in_ver=='v2':
                conv1_1 = tf.layers.conv2d( inputs=data, padding='same', kernel_size=(3, 3), strides=(1, 1), filters=filters[0], 
                                            kernel_initializer=initializer, use_bias=False, name='stem_conv1')
            bn1_2 = tf.layers.batch_normalization(inputs=conv1_1, epsilon=bnEps, momentum=bnMom, training=training, name='stem_bn2')
            act1_2 = TfResNet.Act(data=bn1_2, act_type="prelu", name='stem_relu1')
            #pool1 = mx.sym.Pooling(data=act1_2, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))
            body = act1_2

            # loop over the number of stages
            stride = (2, 2)
            for i in range(0, len(stages)):
                # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume            
                body = residual_module( body, filters[i + 1], stride, training, initializer=initializer, red=True, bnEps=bnEps, bnMom=bnMom, 
                                        bottle_neck=bottle_neck, use_se=use_se, scope='stage%d_unit%d' % (i+1, 1))

                # loop over the number of layers in the stage
                for j in range(0, stages[i] - 1):
                    # apply a ResNet module
                    body = residual_module( body, filters[i + 1], (1, 1), training, initializer=initializer, bnEps=bnEps, bnMom=bnMom, 
                                            bottle_neck=bottle_neck, use_se=use_se, scope='stage%d_unit%d' % (i+1, j+2))

            bn2_1 = tf.layers.batch_normalization(inputs=body, epsilon=bnEps, momentum=bnMom, training=training, name='out_bn2')
            act2_1 = TfResNet.Act(data=bn2_1, act_type="prelu", name='out_relu2')
            conv2_1 = tf.layers.conv2d( inputs=act2_1, padding='same', kernel_size=(3, 3), strides=(1, 1), filters=filters[-1], 
                                        kernel_initializer=initializer, use_bias=False, name='out_conv1')
            # apply BN => ACT => POOL
            bn3_1 = tf.layers.batch_normalization(inputs=conv2_1, epsilon=bnEps, momentum=bnMom, training=training, name='out_bn3')
            act3_1 = TfResNet.Act(data=bn3_1, act_type="relu", name='out_relu3')

            # embedding
            #flatten = mx.sym.Flatten(data=act2_1, name='out_ft1') # don't use flatten here, the parameter will be much bigger
            flt1 = tf.keras.layers.Flatten(name='out_flt1')(act3_1)
            fc1 = tf.layers.dense(inputs=flt1, units=feature_size, kernel_initializer=initializer, name='out_fc1')
            embedding = tf.layers.batch_normalization(inputs=fc1, epsilon=bnEps, momentum=bnMom, training=training, name='out_embedding')

            # return the network architecture
            return embedding, act2_1


    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    # for 'arc': loss_config = {'Arc_margin_scale': 64.0, 'Arc_margin_angle': 0.5}
    # for 'softmax': loss_config = {'Regularizer': tf.keras.regularizers.l2(5e-4), 'Activation': tf.nn.relu}
    @staticmethod
    def build_with_loss(data, label, classes, feature_size, stages, filters, training, loss_config, initializer=None, reuse=None,
                        res_ver='v2', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False, scope=None):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'

        embeddings, _ = TfResNet.build_embedding(   data, feature_size, stages, filters, training, initializer=initializer, res_ver=res_ver, in_ver=in_ver, 
                                                    bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, reuse=reuse, scope=scope)

        with tf.variable_scope(scope, reuse=reuse):        
            if loss_config['loss_type']=='arc':
                logit, inference_loss = TfLosses.arc_loss(embedding=embeddings, labels=label, w_init=initializer, 
                                                          out_num=classes, s=loss_config['Arc_margin_scale'], 
                                                          m=loss_config['Arc_margin_angle'])
            elif loss_config['loss_type']=='softmax':
                logit, inference_loss = TfLosses.softmax_loss(embedding=embeddings, labels=label, out_num=classes,  
                                                              act=loss_config['Activation'], reg=loss_config['Regularizer'], 
                                                              init=initializer)
            pred = tf.nn.softmax(logit, name='prediction') # output name: 'SpoofDenseNet/prediction'
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), label), dtype=tf.float32))

        # return the network architecture
        return embeddings, logit, inference_loss, pred, acc
 
