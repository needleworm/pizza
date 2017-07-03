"""
    A.I. Pizza
    CEO Bhban
    Imagination Garden
    Latest Modification : 6/22, 2017
"""

import tensorflow as tf
import midi2tensor as mt
import numpy as np
import os
import sys
import model as G
import time

__author__ = "BHBAN"


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('mode', "train", "mode : train/ test/ valid [default : train]")
tf.flags.DEFINE_string('device_train', '/cpu:0', "device : /cpu:0 /gpu:0 /gpu:1 [default : /gpu:0]")
tf.flags.DEFINE_string('device_valid', '/cpu:0', "device : /cpu:0 /gpu:0 /gpu:1 [default : /cpu:0]")
tf.flags.DEFINE_string('device_test', '/cpu:0', "device : /cpu:0 /gpu:0 /gpu:1 [default : /cpu:0]")
tf.flags.DEFINE_bool('debug', "False", "debug mode : True/ False [default : True]")
tf.flags.DEFINE_bool('reset', "True", "reset : True/False")
tf.flags.DEFINE_integer('hidden_state_size', "100", "window size. [default : 100]")
tf.flags.DEFINE_integer('predict_size', "10", "window size. [default : 10]")
tf.flags.DEFINE_integer("tr_batch_size", "100", "batch size for training. [default : 100]")
tf.flags.DEFINE_integer("val_batch_size", "1", "batch size for validation. [default : 1]")
tf.flags.DEFINE_integer("test_batch_size", "1", "batch size for validation. [default : 1]")
tf.flags.DEFINE_integer("LSTM_size", "1500", "LSTM size. [default : 1500]")
tf.flags.DEFINE_integer("LSTM_layers", "2", "LSTM size. [default : 2]")
tf.flags.DEFINE_integer("max_grad_norm", "10", "LSTM size. [default : 10]")

logs_dir = "logs"
train_dir = "train_data/"
test_dir = "test_data/"

if FLAGS.mode is 'evaluation':
    FLAGS.reset = False

if FLAGS.reset:
    print("----- The Directory was reset -----")

    if 'win32' in sys.platform or 'win64' in sys.platform:
        os.popen('rmdir /s /q ' + logs_dir)
    else:
        os.popen('rm -rf ' + logs_dir + '/*')
        os.popen('rm -rf ' + logs_dir)

    os.popen('mkdir ' + logs_dir)
    os.popen('mkdir ' + logs_dir + '/out_midi')
    os.popen('mkdir ' + logs_dir + '/train')
    os.popen('mkdir ' + logs_dir + '/valid')
    os.popen('mkdir ' + logs_dir + '/learning_rate')

learning_rate = 1.0
MAX_MAX_EPOCH = 55
MAX_EPOCH = 14
dropout_rate = 0.5
lr_decay = 1/1.15


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {"cost": model.cost, "final_state": model.final_state}

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        costs += cost
        iters += model.prediction_size

        if verbose and step % (model.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):

    #                               Graph Part                                 #
    print("Graph initialization...")
    with tf.device(FLAGS.device_train):
        with tf.name_scope("Train"):
            m_train = G.Model(batch_size = FLAGS.tr_batch_size, hidden_state_size=FLAGS.hidden_state_size,
                              prediction_size=FLAGS.predict_size, is_training=True, lstm_size=FLAGS.LSTM_size,
                              keep_prob=dropout_rate, num_layer=FLAGS.LSTM_layers, max_grad_norm=FLAGS.max_grad_norm)

    with tf.device(FLAGS.device_valid):
        with tf.name_scope("Valid"):
            m_valid = G.Model(batch_size = FLAGS.val_batch_size, hidden_state_size=FLAGS.hidden_state_size,
                              prediction_size=FLAGS.predict_size, is_training=False, lstm_size=FLAGS.LSTM_size,
                              keep_prob=dropout_rate, num_layer=FLAGS.LSTM_layers, max_grad_norm=FLAGS.max_grad_norm)

    with tf.device(FLAGS.device_test):
        with tf.name_scope("Test"):
            m_test = G.Model(batch_size=FLAGS.test_batch_size, hidden_state_size=FLAGS.hidden_state_size,
                              prediction_size=FLAGS.predict_size, is_training=False, lstm_size=FLAGS.LSTM_size,
                              keep_prob=dropout_rate, num_layer=FLAGS.LSTM_layers, max_grad_norm=FLAGS.max_grad_norm)
    print("Done")

    #                               Summary Part                               #
    print("Setting up summary op...")
    tr_loss_ph = tf.placeholder(dtype=tf.float32)
    test_loss_ph = tf.placeholder(dtype=tf.float32)
    learning_rate_ph = tf.placeholder(dtype=tf.float32)
    loss_summary_op = tf.summary.scalar("train_loss", tr_loss_ph)
    loss_summary_op = tf.summary.scalar("test_loss", test_loss_ph)
    learning_rate_summary_op =tf.summary.scalar("learning_rate", learning_rate_ph)
    valid_summary_writer = tf.summary.FileWriter(logs_dir + '/valid/', max_queue=2)
    train_summary_writer = tf.summary.FileWriter(logs_dir + '/train/', max_queue=2)
    learning_rate_summary_writer = tf.summary.FileWriter(logs_dir + '/learning_rate/', max_queue=2)
    print("Done")

    #                               Model Save Part                            #
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")

    #                               Session Part                               #
    print("Setting up Data Reader...")
    if FLAGS.mode == "train":
        train_dataset_reader = mt.Dataset(train_dir, FLAGS.tr_batch_size, FLAGS.hidden_state_size, FLAGS.predict_size)
    validation_dataset_reader = mt.Dataset(test_dir, FLAGS.tr_batch_size, FLAGS.hidden_state_size, FLAGS.predict_size)
    print("done")

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:  # model restore
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization

    if FLAGS.mode == "train":
        for itr in range(MAX_MAX_EPOCH):
            lr_dec = lr_decay ** max(itr + 1 - MAX_EPOCH, 0.0)
            m_train.assign_lr(sess, learning_rate * lr_dec)
            print("Epoch: " + str(itr + 1) + " Learning Rate: " + str(sess.run(m_test.lr)) + ".")

            train_perplexity = run_epoch(sess, m_train, eval_op=m_train.train_op, verbose=True)
            print("Epoch: " + str(itr + 1) + " Train Perplexity: " + str(train_perplexity))
            valid_perplexity = run_epoch(sess, m_valid)
            print("Epoch: " + str(itr + 1) + " Valid Perplexity: " + str(valid_perplexity))

        test_perplexity = run_epoch(sess, m_test)
        print("Test Perplexity: " + str(test_perplexity))

        saver.save(sess, logs_dir + "/model.ckpt", itr)

if __name__ == "__main__":
    tf.app.run()
