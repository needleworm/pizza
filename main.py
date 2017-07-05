"""
    A.I. Pizza
    CTO Bhban
    Imagination Garden
    Latest Modification : 7/3, 2017
"""

import tensorflow as tf
import midi2tensor as mt
import numpy as np
import os
import sys
import model as G
import time
import utils

__author__ = "BHBAN"


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('mode', "train", "mode : train/ test/ valid [default : train]")
tf.flags.DEFINE_string('device_train', '/gpu:0', "device : /cpu:0 /gpu:0 /gpu:1 [default : /gpu:0]")
tf.flags.DEFINE_string('device_valid', '/gpu:0', "device : /cpu:0 /gpu:0 /gpu:1 [default : /cpu:0]")
tf.flags.DEFINE_string('device_test', '/gpu:0', "device : /cpu:0 /gpu:0 /gpu:1 [default : /cpu:0]")
tf.flags.DEFINE_bool('debug', "False", "debug mode : True/ False [default : True]")
tf.flags.DEFINE_bool('reset', "True", "reset : True/False")
tf.flags.DEFINE_bool('use_began_loss', "False", "began loss? : True/False")
tf.flags.DEFINE_integer('hidden_state_size', "300", "window size. [default : 100]")
tf.flags.DEFINE_integer('predict_size', "300", "window size. [default : 10]")
tf.flags.DEFINE_integer("tr_batch_size", "4", "batch size for training. [default : 100]")
tf.flags.DEFINE_integer("val_batch_size", "4", "batch size for validation. [default : 1]")
tf.flags.DEFINE_integer("test_batch_size", "1", "batch size for validation. [default : 1]")
tf.flags.DEFINE_integer("num_keys", "128", "Keys. [default : 88]")

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

learning_rate = 0.0001
MAX_MAX_EPOCH =300000
MAX_EPOCH = 14
dropout_rate = 0.5
lr_decay = 1/1.15
tick_interval = 0.06


def main(_):

    #                               Graph Part                                 #
    print("Graph initialization...")
    with tf.device(FLAGS.device_train):
        with tf.variable_scope("model", reuse=None):
            m_train = G.GAN(batch_size=FLAGS.tr_batch_size,
                            is_training=True,
                            num_keys=FLAGS.num_keys,
                            input_length=FLAGS.hidden_state_size,
                            output_length=FLAGS.predict_size,
                            learning_rate=learning_rate,
                            use_began_loss=FLAGS.use_began_loss)

    with tf.device(FLAGS.device_valid):
        with tf.variable_scope("model", reuse=True):
            m_valid = G.GAN(batch_size=FLAGS.val_batch_size,
                            is_training=False,
                            num_keys=FLAGS.num_keys,
                            input_length=FLAGS.hidden_state_size,
                            output_length=FLAGS.predict_size,
                            learning_rate=learning_rate,
                            use_began_loss=FLAGS.use_began_loss)
    with tf.device(FLAGS.device_test):
        with tf.variable_scope("model", reuse=True):
            m_test = G.GAN(batch_size=FLAGS.test_batch_size,
                            is_training=False,
                            num_keys=FLAGS.num_keys,
                            input_length=FLAGS.hidden_state_size,
                            output_length=FLAGS.predict_size,
                            learning_rate=learning_rate,
                            use_began_loss=FLAGS.use_began_loss)
    print("Done")

    #                               Summary Part                               #
    print("Setting up summary op...")
    g_loss_ph = tf.placeholder(dtype=tf.float32)
    d_loss_ph = tf.placeholder(dtype=tf.float32)
    loss_summary_op_d = tf.summary.scalar("discriminatr_loss", d_loss_ph)
    loss_summary_op_g = tf.summary.scalar("generator_loss", g_loss_ph)
    valid_summary_writer = tf.summary.FileWriter(logs_dir + '/valid/', max_queue=2)
    train_summary_writer = tf.summary.FileWriter(logs_dir + '/train/', max_queue=2)
    print("Done")

    #                               Model Save Part                            #
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")

    #                               Session Part                               #
    print("Setting up Data Reader...")
    validation_dataset_reader = mt.Dataset(test_dir, FLAGS.tr_batch_size, FLAGS.hidden_state_size, FLAGS.predict_size, FLAGS.num_keys, tick_interval)
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
        train_dataset_reader = mt.Dataset(train_dir, FLAGS.tr_batch_size, FLAGS.hidden_state_size, FLAGS.predict_size, FLAGS.num_keys, tick_interval)
        for itr in range(MAX_MAX_EPOCH):
            feed_dict = utils.run_epoch(train_dataset_reader, FLAGS.tr_batch_size, m_train, sess, dropout_rate, began_loss=FLAGS.use_began_loss)

            if itr % 10 == 0:
                if FLAGS.use_began_loss:
                    train_loss_d, train_loss_g, train_pred = sess.run([m_train.loss, m_train.loss_g, m_train.predict], feed_dict = feed_dict)
                    train_summary_str_d, train_summary_str_g = sess.run([loss_summary_op_d, loss_summary_op_g],
                                                                    feed_dict={g_loss_ph: train_loss_g,
                                                                               d_loss_ph: train_loss_d})
                    train_summary_writer.add_summary(train_summary_str_g , itr)
                    print("Step : %d  TRAINING LOSS *****************" %(itr))
                    print("Dicriminator_loss: %g\nGenerator_loss: %g" %(train_loss_d, train_loss_g))
                else:
                    train_loss, train_pred = sess.run([m_train.loss, m_train.predict], feed_dict=feed_dict)
                    train_summary_str_d = sess.run(loss_summary_op_d, feed_dict={d_loss_ph:train_loss})
                    train_summary_writer.add_summary(train_summary_str_d, itr)
                    print("Step : %d  TRAINING LOSS %g" %(itr, train_loss))
                    

            if itr % 100 == 0:
                if FLAGS.use_began_loss:
                    valid_loss_d, valid_loss_g, valid_pred = utils.validation(validation_dataset_reader,
                                                                              FLAGS.val_batch_size, m_valid,
                                                                              FLAGS.hidden_state_size,
                                                                              FLAGS.predict_size, sess,
                                                                              logs_dir, itr, tick_interval, FLAGS.use_began_loss)
                    valid_summary_str_d, valid_summary_str_g = sess.run([loss_summary_op_d, loss_summary_op_g],
                                                                         feed_dict={g_loss_ph: valid_loss_g,
                                                                                    d_loss_ph: valid_loss_d})
                    valid_summary_writer.add_summary(valid_summary_str_d, itr)
                    print("Step : %d  VALIDATION LOSS ***************" %(itr))
                    print("Dicriminator_loss: %g\nGenerator_loss: %g" %(valid_loss_d, valid_loss_g))
                else:
                    valid_loss, valid_pred = utils.validation(validation_dataset_reader,
                                                              FLAGS.val_batch_size, m_valid,
                                                              FLAGS.hidden_state_size,
                                                              FLAGS.predict_size, sess,
                                                              logs_dir, itr, tick_interval, FLAGS.use_began_loss)
                    valid_summary_str_d = sess.run(loss_summary_op_d, feed_dict={d_loss_ph:train_loss})
                    valid_summary_writer.add_summary(valid_summary_str_d, itr)
                    print("Step : %d  VALIDATION LOSS %g" %(itr, valid_loss))

            if itr % 1000 == 0:
                saver.save(sess, logs_dir + "/model.ckpt", itr)

if __name__ == "__main__":
    tf.app.run()
