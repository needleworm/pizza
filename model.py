"""
    A.I. Pizza
    CEO Bhban
    Imagination Garden
    Latest Modification : 6/22, 2017
"""

import tensorflow as tf
import inspect

__author__ = "BHBAN"


class Model(object):
    def __init__(self, batch_size, hidden_state_size, prediction_size, is_training, lstm_size, keep_prob, num_layer,
                 max_grad_norm, name):
        self.input = tf.placeholder(tf.float32, [None, hidden_state_size, 88])
        self.ground_truth = tf.placeholder(tf.int32, [None, prediction_size])
        #embedding = tf.get_variable("embedding", [88, lstm_size], dtype=tf.float32)
        #inputs = tf.nn.embedding_lookup(embedding, self.input)

        self.epoch_size = ((hidden_state_size // batch_size) -1) // prediction_size
        self.prediction_size = prediction_size
        self.batch_size = batch_size

        if is_training and keep_prob < 1:
            self.input = tf.nn.dropout(self.input, keep_prob)

        outputs = []

        self.network = Graph(batch_size, hidden_state_size, is_training, lstm_size, num_layer)
        self.cell, self.initial_state = self.network.graph(self.input, keep_prob)
        state = self.initial_state

        with tf.variable_scope("RNN"):
            for time_step in range(prediction_size):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(self.input[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, lstm_size])

        softplus_W = tf.get_variable(name + "w", [lstm_size, 88], dtype=tf.float32)
        softplus_b = tf.get_variable(name + "b", [88], dtype=tf.float32)
        logits = tf.nn.bias_add(tf.matmul(output, softplus_W), softplus_b)

        self.logits = tf.reshape(logits, [batch_size, prediction_size, 88])
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.ground_truth,
                                                     tf.ones([batch_size, prediction_size], dtype=tf.float32),
                                                     average_across_timesteps=False,
                                                     average_across_batch=True)

        self.cost = cost = tf.reduce_sum(self.loss)
        self.final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainable_vars), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_vars),
                                                  global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[])
        self._lr_update = tf.assign(self._lr, self._new_lr)

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

        initial_state = cell.zero_state(self.batch_size, tf.float32)
        return cell, initial_state


