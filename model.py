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

class GAN(object):
    def __init__(self, batch_size, is_training, num_keys, input_length, output_length, learning_rate, use_began_loss=False):
        self.keep_probability = tf.placeholder(tf.float16, name="keep_probability")
        self.input_music_seg = tf.placeholder(tf.float16, shape=[batch_size, num_keys, input_length, 1], name="input_music_segment")
        self.ground_truth_seg = tf.placeholder(tf.float16, shape=[batch_size, num_keys, output_length, 1], name="ground_truth")

        self.Generator = Generator(is_training)
        self.Discriminator = Discriminator(is_training)

        with tf.variable_scope("G"):        
            self.predict, logits = self.Generator.predict(self.input_music_seg, is_training, self.keep_probability, num_keys, output_length)
        with tf.variable_scope("D") as discriminator_scope:
            self.d_out1, d_logits1 = self.Discriminator.discriminate(self.ground_truth_seg, is_training, self.keep_probability)
            discriminator_scope.reuse_varialbes()
            self.d_out2, d_logits2 = self.Discriminator.discriminate(self.predict, is_training, self.keep_probability)
            
        # basic loss
        self.loss = tf.reduce_mean(-tf.log(d_logits1) - tf.log(1-d_logits2))
        # began loss
        if use_began_loss:
            self.loss_g = tf.reduce_sum(tf.squared_difference(self.ground_truth_seg, self.predict))
            
        trainable_var = tf.trainable_variables()
        
        if use_began_loss:
            self.train_op, self.train_op_g = self.train_with_began_loss(trainable_var, learning_rate)
        else:
            self.train_op = self.train_without_began_loss(trainable_var, learning_rate)
            
    def train_with_began_loss(self, trainable_var, learning_rate):
        optimizer1 = tf.train.AdamOptimizer(learning_rate)
        optimizer2 = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer1.compute_gradient(self.loss, var_list = trainable_var)
        grads_g = optimizer2.compute_gradient(self.loss_g, var_list = trainable_var)
        return optimizer1.apply_gradients(grads), optimizer2.apply_gradients(grads_g)
    
    def train_without_began_loss(self, trainable_ar, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradient(self.loss, var_list = trainable_var)
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
        
        self.CNN_shapes.append([16, 1, 1, 256])
        self.CNN_shapes.append([2, 2, 256, 256])
        self.CNN_shapes.append([2, 2, 256, 512])
        self.CNN_shapes.append([2, 2, 512, 512])
        self.CNN_shapes.append([2, 2, 512, 512])
        
        for i. el in enumerate(self.CNN_shapes):
            self.CNN_kernels.append(tf.get_variable("E_CNN_" + str(i), initializer=tf.truncated_normal(el, stddev=0.02)))
       
    def predict(self, input_music, is_training, keep_prob, num_keys, output_length):
        net = []
        net.append(input_music)
        deconv_shapes = []
        dcnn_kernels = []

        # Encoder Layers
        for i, el in enumerate(self.CNN_kernels):
            C = tf.nn.conv2d(net[-1], el, strides=[1, 1, 1, 1], padding="SAME")
            N = tf.contrib.layers.batch_norm(C, decay=decay, is_training=is_training, update_collections=None)
            R = tf.nn.relu(N)
            net.append(R)
            
        # Decoder Layers
        for el in net:
            deconv_shape.append(el.shape.as_list())
            
        for i in range(len(deconv_shape)):
            dcnn_shape = deconv_shape[-1-i]
            dcnn_shape[2] = dcnn_shape[3]
            dcnn_shape[3] = net[-i -1].get_shape().as_list()[3]]
            dcnn_kernels.append(tf.get_variable("D_DCNN_" + str(i), initializer=tf.truncated_normal(dcnn_shape, stddev=0.02)))
        
        for i in range(len(dcnn_kernels)):
            DC = tf.nn.conv2d_transpose(net[-i-1], dcnn_kernels[-i-1], deconv_shaape[i], strides=[1, 1, 1, 1], padding="SAME")
            F = tf.add(DC, net[-i-1])
            net.append(F)
        
        logits = net[-1]
        predict = tf.round(logits)
        
    return predict, logits


class Discriminator(object):
    def __init__(self, is_training):
        self.is_training = is_training
        self.cnn_shapes = []
        self.cnn_kernels = []
        self.fnn_shapes = []
        self.fnn_kernels = []
        
        self.cnn_shapes.append([2, 2, 3, 64])
        self.cnn_shapes.append([2, 2, 64, 128])
        self.cnn_shapes.append([2, 2, 128, 256])
        self.cnn_shapes.append([2, 2, 256, 512])
        self.cnn_shapes.append([2, 2, 512, 512])
        
        self.fnn_shapes.append([512, 4096])
        self.fnn_shapes.append([4096, 4096])
        self.fnn_shapes.append([4096, 1024])
        self.fnn_shapes.append([1024, 2])
        
        for i, el in enumerate(self.cnn_shapes):
            self.CNN_kernels.append(tf.get_variable("D_CNN_" + str(i), initializer=tf.truncated_normal(el, stddev=0.02)))
        
        for i, el in enumerate(self.fnn_shapes):
            self.fnn_kernels.append(tf.get_variables("D_FNN_" + str(i), initializer = tf.truncated_normal(el, stddev=0.02)))
        
    def discriminate(self, input_music, is_training, keep_prob):
        net = []
        net.append(input_music)
        
        for el in self.cnn_kernels:
            C = tf.nn.conv2d(net[-1], el, strides=[1,1,1,1], padding="SAME")
        """START_HERE"""
        
        

class Model(object):
    def __init__(self, batch_size, hidden_state_size, prediction_size, is_training, lstm_size, keep_prob, num_layer,
                 max_grad_norm):
        self.input = tf.placeholder(tf.int64, [None, hidden_state_size, 88])
        self.ground_truth = tf.placeholder(tf.float16, [None, prediction_size, 88])
        embedding = tf.get_variable("embedding", [88, hidden_state_size], dtype=tf.float16)
        inputs = tf.nn.embedding_lookup(embedding, self.input)

        self.epoch_size = ((hidden_state_size // batch_size) -1) // prediction_size
        self.prediction_size = prediction_size
        self.batch_size = batch_size

        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        outputs = []

        with tf.variable_scope("RNN"):
            self.network = Graph(batch_size, hidden_state_size, is_training, lstm_size, num_layer)
            self.cell, self.initial_state = self.network.graph(self.input, keep_prob)
            state = self.initial_state
            for time_step in range(prediction_size):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, lstm_size])

        softplus_W = tf.get_variable("softplus_W", [lstm_size, 88], dtype=tf.float16)
        softplus_b = tf.get_variable("softplus_b", [88], dtype=tf.float16)
        logits = tf.nn.bias_add(tf.matmul(output, softplus_W), softplus_b)

        self.logits = tf.reshape(logits, [batch_size, prediction_size, 88])
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.ground_truth,
                                                     tf.ones([batch_size, prediction_size], dtype=tf.float16),
                                                     average_across_timesteps=False,
                                                     average_across_batch=True)

        self.cost = cost = tf.reduce_sum(self.loss)
        self.final_state = state

        if not is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainable_vars), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_vars),
                                                  global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float16, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


class Graph:
    def __init__(self, batch_size, hidden_state_size, is_training, lstm_size, num_layer):
        self.batch_size = batch_size
        self.hidden_state_size = hidden_state_size
        self.is_training = is_training
        self.lstm_size = lstm_size
        self.num_layer = num_layer

    def graph(self, input, keep_prob):
        def lstm_cell():
            if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=0.0, state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        if self.is_training and keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.num_layer)], state_is_tuple=True)

        initial_state = cell.zero_state(self.batch_size, tf.float16)
        return cell, initial_state


