"""
    A.I. Pizza
    CEO Bhban
    Imagination Garden
    Latest Modification : 6/22, 2017
"""

import tensorflow as tf
import inspect

__author__ = "BHBAN"

decay = 0.9


class BEGAN(object):
    def __init__(self, batch_size, is_training, num_keys, input_length, output_length, learning_rate):
        self.input_music_seg = tf.placeholder(tf.float32, shape=[batch_size, 8, num_keys/8, input_length], name="input_music_segment")
        self.ground_truth_seg = tf.placeholder(tf.float32, shape=[batch_size, 8, num_keys/8, output_length], name="ground_truth")

        self.GAN = Generator_BEGAN(is_training, input_length, output_length)
        
        self.k_t = tf.Variable(0., trainable=False, name='k_t')
        self.lambda_k = learning_rate

        with tf.variable_scope("graph"):
            self.predict, g_logits = self.GAN.predict(self.input_music_seg, is_training)
        with tf.variable_scope("graph", reuse=True):
            self.d_out1, d_logits = self.GAN.predict(self.predict, is_training)
            
        self.l_x = tf.reduce_mean(tf.abs(d_logits - self.ground_truth_seg))
        self.l_g = tf.reduce_mean(tf.abs(g_logits - self.ground_truth_seg))
        
        self.loss_d = self.l_x - self.k_t * self.l_g
        self.loss_g = self.l_g
        
        self.train_op_d, self.train_op_g=self.train(learning_rate)
        
    def train(self, learning_rate):
        optimizer_d = tf.train.AdamOptimizer(learning_rate)
        optimizer_g = tf.train.AdamOptimizer(learning_rate)
        train_op_d = optimizer_d.minimize(self.loss_d)
        train_op_g = optimizer_g.minimize(self.loss_g) 
        gamma = self.l_x / self.l_g
        
        self.k_t += self.lambda_k*(gamma * self.l_x - self.l_g) 
        
        return train_op_d, train_op_g
 
    
class Generator_BEGAN(object):
    def __init__(self, is_training, input_window, output_window):
        self.window_size = input_window
        self.output_window = output_window
        self.is_training = is_training
        self.CNN_shapes = []
        self.CNN_kernels = []

        self.CNN_shapes.append([2, 2, input_window, 64])
        self.CNN_shapes.append([2, 2, 64, 128])
        self.CNN_shapes.append([2, 2, 128, 256])
        self.CNN_shapes.append([2, 2, 256, 256])
        self.CNN_shapes.append([2, 2, 256, 256])
        self.CNN_shapes.append([1, 1, 256, 256])

        for i, el in enumerate(self.CNN_shapes):
            self.CNN_kernels.append(tf.get_variable("G_CNN_" + str(i), initializer=tf.truncated_normal(el, stddev=0.02)))

    def predict(self, input_music, is_training):
        net = []
        net.append(input_music)
        dcnn_kernels = []

        # Encoder Layers
        for i, el in enumerate(self.CNN_kernels):
            C = tf.nn.conv2d(net[-1], el, strides=[1, 2, 2, 1], padding="SAME")
            N = tf.contrib.layers.batch_norm(C, decay=decay, is_training=is_training, updates_collections=None)
            R = tf.nn.relu(N)
            net.append(R)

        # Decoder Layers
        deconv_shape1 = net[6].shape.as_list()
        dcnn1_shape = [1, 1, deconv_shape1[3], net[-1].get_shape().as_list()[3]]
        dcnn_kernels.append(tf.get_variable("DCNN_1_W", initializer=tf.truncated_normal(dcnn1_shape, stddev=0.02)))

        deconv_shape2 = net[5].shape.as_list()
        dcnn2_shape = [2, 2, deconv_shape2[3], deconv_shape1[3]]
        dcnn_kernels.append(tf.get_variable("DCNN_2_W", initializer=tf.truncated_normal(dcnn2_shape, stddev=0.02)))

        deconv_shape3 = net[4].shape.as_list()
        dcnn3_shape = [2, 2, deconv_shape3[3], deconv_shape2[3]]
        dcnn_kernels.append(tf.get_variable("DCNN_3_W", initializer=tf.truncated_normal(dcnn3_shape, stddev=0.02)))
        
        deconv_shape4 = net[3].shape.as_list()
        dcnn4_shape = [2, 2, deconv_shape4[3], deconv_shape3[3]]
        dcnn_kernels.append(tf.get_variable("DCNN_4_W", initializer=tf.truncated_normal(dcnn4_shape, stddev=0.02)))

        deconv_shape5 = net[2].shape.as_list()
        dcnn5_shape = [2, 2, deconv_shape5[3], deconv_shape4[3]]
        dcnn_kernels.append(tf.get_variable("DCNN_5_W", initializer=tf.truncated_normal(dcnn5_shape, stddev=0.02)))

        deconv_shape6 = net[1].shape.as_list()
        dcnn6_shape = [2, 2, deconv_shape6[3], deconv_shape5[3]]
        dcnn_kernels.append(tf.get_variable("DCNN_6_W", initializer=tf.truncated_normal(dcnn6_shape, stddev=0.02)))

        deconv_shape7 = net[0].shape.as_list()
        deconv_shape7[3] = self.output_window
        dcnn7_shape = [2, 2, self.output_window, deconv_shape6[3]]
        dcnn_kernels.append(tf.get_variable("DCNN_7_W", initializer=tf.truncated_normal(dcnn7_shape, stddev=0.02)))

        DC1 = tf.nn.conv2d_transpose(net[-1], dcnn_kernels[0], deconv_shape1, strides=[1,2,2,1], padding="SAME")
        DC1 = tf.contrib.layers.batch_norm(DC1, decay=decay, is_training=is_training, updates_collections=None)
        F1 = tf.add(DC1, net[6], name="f1")

        DC2 = tf.nn.conv2d_transpose(DC1, dcnn_kernels[1], deconv_shape2, strides=[1,2,2,1], padding="SAME")
        DC2 = tf.contrib.layers.batch_norm(DC2, decay=decay, is_training=is_training, updates_collections=None)
        F2 = tf.add(DC2, net[5], name="F2")

        DC3 = tf.nn.conv2d_transpose(DC2, dcnn_kernels[2], deconv_shape3, strides=[1,2,2,1], padding="SAME")
        DC3 = tf.contrib.layers.batch_norm(DC3, decay=decay, is_training=is_training, updates_collections=None)
        F3 = tf.add(DC3, net[4], name="F3")

        DC4 = tf.nn.conv2d_transpose(DC3, dcnn_kernels[3], deconv_shape4, strides=[1,2,2,1], padding="SAME")
        DC4 = tf.contrib.layers.batch_norm(DC4, decay=decay, is_training=is_training, updates_collections=None)
        F4 = tf.add(DC4, net[3], name="F4")

        DC5 = tf.nn.conv2d_transpose(DC4, dcnn_kernels[4], deconv_shape5, strides=[1,2,2,1], padding="SAME")
        DC5 = tf.contrib.layers.batch_norm(DC5, decay=decay, is_training=is_training, updates_collections=None)
        F5 = tf.add(DC5, net[2], name="F5")
        
        DC6 = tf.nn.conv2d_transpose(DC5, dcnn_kernels[5], deconv_shape6, strides=[1,2,2,1], padding="SAME")
        DC6 = tf.contrib.layers.batch_norm(DC6, decay=decay, is_training=is_training, updates_collections=None)
        F6 = tf.add(DC6, net[1], name="F6")
        
        DC7 = tf.nn.conv2d_transpose(DC6, dcnn_kernels[6], deconv_shape7, strides=[1,2,2,1], padding="SAME")
        DC7 = tf.contrib.layers.batch_norm(DC7, decay=decay, is_training=is_training, updates_collections=None)
        F7 = tf.add(DC7, net[0], name="F7")


        logits = F7

        predict = tf.nn.sigmoid(F7)

        return predict, logits
    

class VAE(object):
    def __init__(self, batch_size, is_training, num_keys, input_length, output_length, learning_rate):
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")
        self.input_music_seg = tf.placeholder(tf.float32, shape=[batch_size, num_keys, input_length, 1], name="input_music_segment")
        self.ground_truth_seg = tf.placeholder(tf.float32, shape=[batch_size, num_keys, output_length, 1], name="ground_truth")
        self.Generator = Generator(is_training)
        
        self.predict, logits = self.Generator.predict(self.input_music_seg, is_training, self.keep_probability, num_keys, output_length)
        
        self.loss = tf.reduce_mean(tf.squared_difference(self.ground_truth_seg, logits))
        
        trainable_var = tf.trainable_variables()
        self.train_op = self.train(trainable_var, learning_rate)
        
    def train(self, trainable_var, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        return optimizer.apply_gradients(grads)


class GAN(object):
    def __init__(self, batch_size, is_training, num_keys, input_length, output_length, learning_rate, use_began_loss=False):
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")
        self.input_music_seg = tf.placeholder(tf.float32, shape=[batch_size, num_keys, input_length, 1], name="input_music_segment")
        self.ground_truth_seg = tf.placeholder(tf.float32, shape=[batch_size, num_keys, output_length, 1], name="ground_truth")

        self.Generator = Generator(is_training)
        self.Discriminator = Discriminator(is_training)

        with tf.variable_scope("G"):
            self.predict, logits = self.Generator.predict(self.input_music_seg, is_training, self.keep_probability, num_keys, output_length)
        with tf.variable_scope("D") as discriminator_scope:
            self.d_out1, d_logits1 = self.Discriminator.discriminate(self.ground_truth_seg, is_training, self.keep_probability)
            #discriminator_scope.reuse_variables()
            self.d_out2, d_logits2 = self.Discriminator.discriminate(self.predict, is_training, self.keep_probability)

        # basic loss
        self.loss = tf.reduce_mean(-tf.log(d_logits1) - tf.log(1-d_logits2))
        # began loss
        if use_began_loss:
            self.loss_g = tf.reduce_mean(tf.squared_difference(self.ground_truth_seg, logits))

        trainable_var = tf.trainable_variables()

        if use_began_loss:
            self.train_op, self.train_op_g = self.train_with_began_loss(trainable_var, learning_rate)
        else:
            self.train_op = self.train_without_began_loss(trainable_var, learning_rate)

    def train_with_began_loss(self, trainable_var, learning_rate):
        optimizer1 = tf.train.AdamOptimizer(learning_rate)
        optimizer2 = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer1.compute_gradients(self.loss, var_list = trainable_var)
        grads_g = optimizer2.compute_gradients(self.loss_g, var_list = trainable_var)
        return optimizer1.apply_gradients(grads), optimizer2.apply_gradients(grads_g)

    def train_without_began_loss(self, trainable_var, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss, var_list = trainable_var)
        return optimizer.apply_gradients(grads)

    def discrimination(self, itr):
        if self.d_out1 == self.d_out2:
            print("EPOCH : " + str(itr) + " >>> Discriminator Failed!!!!!! Sibbal!!")
        else:
            print("EPOCH : " + str(itr) + " >>> Discriminator successed!!!!!!")


class Generator(object):
    def __init__(self, is_training):
        self.is_training = is_training
        self.CNN_shapes = []
        self.CNN_kernels = []

        self.CNN_shapes.append([16, 16, 1, 32])
        self.CNN_shapes.append([8, 8, 32, 64])
        self.CNN_shapes.append([8, 8, 64, 64])
        self.CNN_shapes.append([1, 1, 64, 1024])

        for i, el in enumerate(self.CNN_shapes):
            self.CNN_kernels.append(tf.get_variable("E_CNN_" + str(i), initializer=tf.truncated_normal(el, stddev=0.02)))

    def predict(self, input_music, is_training, keep_prob, num_keys, output_length):
        net = []
        net.append(input_music)
        dcnn_kernels = []

        # Encoder Layers
        for i, el in enumerate(self.CNN_kernels):
            C = tf.nn.conv2d(net[-1], el, strides=[1, 2, 2, 1], padding="VALID")
            N = tf.contrib.layers.batch_norm(C, decay=decay, is_training=is_training, updates_collections=None)
            R = tf.nn.relu(N)
            net.append(R)

        # Decoder Layers
        deconv_shape1 = net[3].shape.as_list()
        dcnn1_shape = [1, 1, deconv_shape1[3], net[-1].get_shape().as_list()[3]]
        dcnn_kernels.append(tf.get_variable("D_DCNN_1_W", initializer=tf.truncated_normal(dcnn1_shape, stddev=0.02)))

        deconv_shape2 = net[2].shape.as_list()
        dcnn2_shape = [8, 8, deconv_shape2[3], deconv_shape1[3]]
        dcnn_kernels.append(tf.get_variable("D_DCNN_2_W", initializer=tf.truncated_normal(dcnn2_shape, stddev=0.02)))

        deconv_shape3 = net[1].shape.as_list()
        dcnn3_shape = [8, 8, deconv_shape3[3], deconv_shape2[3]]
        dcnn_kernels.append(tf.get_variable("D_DCNN_3_W", initializer=tf.truncated_normal(dcnn3_shape, stddev=0.02)))

        deconv_shape4 = net[0].shape.as_list()
        dcnn4_shape = [16, 16, deconv_shape4[3], deconv_shape3[3]]
        dcnn_kernels.append(tf.get_variable("D_DCNN_4_W", initializer=tf.truncated_normal(dcnn4_shape, stddev=0.02)))

        DC1 = tf.nn.conv2d_transpose(net[-1], dcnn_kernels[0], deconv_shape1, strides=[1,2,2,1], padding="VALID")
        DC1 = tf.contrib.layers.batch_norm(DC1, decay=decay, is_training=is_training, updates_collections=None)
#        F1 = tf.add(DC1, net[3])

        DC2 = tf.nn.conv2d_transpose(DC1, dcnn_kernels[1], deconv_shape2, strides=[1,2,2,1], padding="VALID")
        DC2 = tf.contrib.layers.batch_norm(DC2, decay=decay, is_training=is_training, updates_collections=None)
#        F2 = tf.add(DC2, net[2])

        DC3 = tf.nn.conv2d_transpose(DC2, dcnn_kernels[2], deconv_shape3, strides=[1,2,2,1], padding="VALID")
        DC3 = tf.contrib.layers.batch_norm(DC3, decay=decay, is_training=is_training, updates_collections=None)
#        F3 = tf.add(DC3, net[1])

        DC4 = tf.nn.conv2d_transpose(DC3, dcnn_kernels[3], deconv_shape4, strides=[1,2,2,1], padding="VALID")
        DC4 = tf.contrib.layers.batch_norm(DC4, decay=decay, is_training=is_training, updates_collections=None)
#        F4 = tf.add(DC4, net[0])

        logits = DC4

        predict = tf.round(logits)

        return predict, logits


class Discriminator(object):
    def __init__(self, is_training):
        self.is_training = is_training
        self.CNN_shapes = []
        self.CNN_kernels = []
        self.FNN_shapes = []
        self.FNN_kernels = []
        self.FNN_biases = []

        self.CNN_shapes.append([2, 2, 1, 64])
        self.CNN_shapes.append([2, 2, 64, 128])
        self.CNN_shapes.append([2, 2, 128, 256])
        self.CNN_shapes.append([2, 2, 256, 512])
        self.CNN_shapes.append([2, 2, 512, 512])

        self.FNN_shapes.append([512, 4096])
        self.FNN_shapes.append([4096, 4096])
        self.FNN_shapes.append([4096, 1024])
        self.FNN_shapes.append([1024, 2])

        for i, el in enumerate(self.CNN_shapes):
            self.CNN_kernels.append(tf.get_variable("D_CNN_" + str(i), initializer=tf.truncated_normal(el, stddev=0.02)))

        for i, el in enumerate(self.FNN_shapes):
            self.FNN_kernels.append(tf.get_variable("D_FNN_" + str(i), initializer = tf.truncated_normal(el, stddev=0.02)))

    def discriminate(self, input_music, is_training, keep_prob):
        net = []
        net.append(input_music)

        for el in self.CNN_kernels:
            C = tf.nn.conv2d(net[-1], el, strides=[1,1,1,1], padding="SAME")
            N = tf.contrib.layers.batch_norm(C, decay=decay, is_training=is_training, updates_collections=None)
            R = tf.nn.relu(N)
            P = tf.nn.max_pool(R, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding="SAME")
            net.append(P)

        net[-1] = tf.reshape(net[-1], [-1, self.FNN_shapes[0][0]])

        for i, el in enumerate(self.FNN_kernels[:-1]):
            W = tf.matmul(net[-1], el)
            N = tf.contrib.layers.batch_norm(W, is_training=is_training, updates_collections=None)
            R = tf.nn.relu(N)
            net.append(R)

        logits = tf.nn.softmax(net[-1])
        discrimination = tf.argmax(logits)

        return discrimination, logits
