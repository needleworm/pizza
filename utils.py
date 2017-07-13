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

    
def vae_run_epoch(dataset, batch_size, model, session, dropout_rate):
    hidden_state, ground_truth = dataset.next_batch()
    feed_dict = {model.input_music_seg: hidden_state,
                 model.ground_truth_seg: ground_truth,
                 model.keep_probability: dropout_rate}
    
    session.run(model.train_op, feed_dict=feed_dict)
    
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


def vae_validation(dataset, batch_size, model, hidden_state_size, predict_size, session, logs_dir, idx, tick_interval):
    hidden_state, ground_truth = dataset.next_batch()
    feed_dict = {model.input_music_seg: hidden_state,
                 model.ground_truth_seg: ground_truth,
                 model.keep_probability: 1.0}

    loss, predict = session.run([model.loss, model.predict], feed_dict=feed_dict)

    save_music(hidden_state, predict, logs_dir + "/out_midi/", "VALIDATION_MUSICS_" + str(idx).zfill(5), batch_size,
               tick_interval)

    return loss, predict
        

def recursive_validation(line, batch_size, model, hidden_state_size, predict_size, session, logs_dir, idx,
                         tick_interval):
    hidden_state = line[-hidden_state_size:]
    feed_dict = {model.input_music_seg: hidden_state, model.keep_probability: 1.0}

    predict = session.run([model.loss, model.predict], feed_dict=feed_dict)

    save_music(hidden_state, predict, logs_dir + "/out_midi/", "REC_VALIDATION_MUSICS_" + str(idx).zfill(5), batch_size,
               tick_interval)

    line.extend(predict)

    return line
    
    
def test_model(dataset, batch_size, model, predict_size, session, logs_dir, idx, tick_interval, repetition):
    hidden_state, _ = dataset.next_batch()
    batch_size, num_keys, hidden_state_size, _ = hidden_state.shape
    template = np.zeros((batch_size, num_keys, hidden_state_size + predict_size * repetition, 1))
    template[:, :, 0:hidden_state_size, :] = hidden_state
    read_start = 0
    write_start = hidden_state_size
    path = logs_dir + "/out_midi/"

    for i in range(repetition):
        feed_dict = {model.input_music_seg : template[:, :, read_start:read_start + hidden_state_size, :],
                     model.keep_probability:1.0}
        predict = session.run(model.predict, feed_dict=feed_dict)
        write_end = write_start + predict_size
        template[:, :, write_start:write_end, :] = predict
        write_start += predict_size
        read_start += predict_size

    for i in range(batch_size):
        tensor2midi.save_tensor_to_midi(template[i], path + "TEST_MUSIC_" + str(i) + "_" + str(idx), tick_interval)
    print("****************************************************** ")
    print("                   TEST MUSIC SAVED                    ")
    print("****************************************************** ")        


def save_music(hidden_state, predict, path, name, batch_size, tick_interval):
    merged = np.concatenate((hidden_state, predict), axis=2)

    for i in range(batch_size):
        tensor2midi.save_tensor_to_midi(merged[i], path + name + "_" + str(i), tick_interval)
    print("*************** VALIDATION MUSIC SAVED *************** ")
    
    
