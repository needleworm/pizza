"""
    A.I. Pizza
    CTO Bhban
    Imagination Garden
    Latest Modification : 7/3, 2017
"""

import numpy as np

import tensor2midi


def run_epoch(dataset, batch_size, model, session, dropout_rate, began_loss=True):
    hidden_state, ground_truth = dataset.next_batch()
    feed_dict = {model.input_music_seg: hidden_state,
                 model.ground_truth_seg: ground_truth,
                 model.keep_probability: dropout_rate}

    if began_loss:
        train_op, train_op_g = session.run([model.train_op, model.train_op_g], feed_dict=feed_dict)
    else:
        train_op = session.run(model.train_op, feed_dict=feed_dict)

    return feed_dict


def validation(dataset, batch_size, model, hidden_state_size, predict_size, session, logs_dir, idx, tick_interval,
               began_loss=True):
    hidden_state, ground_truth = dataset.next_batch()
    feed_dict = {model.input_music_seg: hidden_state,
                 model.ground_truth_seg: ground_truth,
                 model.keep_probability: 1.0}

    if began_loss:
        loss_d, loss_g, predict = session.run([model.loss, model.loss_g, model.predict], feed_dict=feed_dict)
    else:
        loss_d, predict = session.run([model.loss, model.predict], feed_dict=feed_dict)

    save_music(hidden_state, predict, logs_dir + "/out_midi/", "VALIDATION_MUSICS_" + str(idx).zfill(5), batch_size,
               tick_interval)

    if began_loss:
        return loss_d, loss_g, predict
    else:
        return loss_d, predict


def recursive_validation(line, batch_size, model, hidden_state_size, predict_size, session, logs_dir, idx,
                         tick_interval):
    hidden_state = line[-hidden_state_size:]
    feed_dict = {model.input_music_seg: hidden_state, model.keep_probability: 1.0}

    predict = session.run([model.loss, model.predict], feed_dict=feed_dict)

    save_music(hidden_state, predict, logs_dir + "/out_midi/", "REC_VALIDATION_MUSICS_" + str(idx).zfill(5), batch_size,
               tick_interval)

    line.extend(predict)

    return line


def save_music(hidden_state, predict, path, name, batch_size, tick_interval):
    merged = np.concatenate((hidden_state, predict), axis=2)

    print(merged[0].shape)

    for i in range(batch_size):
        tensor2midi.save_tensor_to_midi(merged[i], path + name + "_" + str(i), tick_interval)
    print("*************** VALIDATION MUSIC SAVED *************** ")
