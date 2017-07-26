"""
    A.I. Pizza
    CTO Bhban
    Imagination Garden
    Latest Modification : 7/3, 2017
"""

import numpy as np

import tensor2midi


def run_epoch(dataset, batch_size, model, session):
    hidden_state, ground_truth = dataset.next_batch_3d()
    feed_dict = {model.input_music_seg: hidden_state,
                 model.ground_truth_seg: ground_truth}

    train_op_d, train_op_g = session.run([model.train_op_d, model.train_op_g], feed_dict=feed_dict)

    return feed_dict

    
def vae_run_epoch(dataset, batch_size, model, session, dropout_rate):
    hidden_state, ground_truth = dataset.next_batch()
    feed_dict = {model.input_music_seg: hidden_state,
                 model.ground_truth_seg: ground_truth,
                 model.keep_probability: dropout_rate}
    
    session.run(model.train_op, feed_dict=feed_dict)
    
    return feed_dict


def validation(dataset, batch_size, model, hidden_state_size, predict_size, session, logs_dir, idx, tick_interval):
    hidden_state, ground_truth = dataset.next_batch_3d()
    feed_dict = {model.input_music_seg: hidden_state,
                 model.ground_truth_seg: ground_truth}

    loss_d, loss_g, pred = session.run([model.loss_d, model.loss_g, model.predict], feed_dict=feed_dict)
    predict = _3d_tensor_to_2d_tensor(pred)

    predict[predict < 0.9] = 0
    predict[predict >= 0.9] = 1

    hidden_state = _3d_tensor_to_2d_tensor(hidden_state)


    save_music(hidden_state, predict, logs_dir + "/out_midi/", "VALIDATION_MUSICS_" + str(idx).zfill(5), batch_size,
               tick_interval)

    return loss_d, loss_g, predict


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
    hidden_state, _ = dataset.next_batch_3d()
    batch_size, octav, reps, hidden_state_size = hidden_state.shape
    template = np.zeros((batch_size, octav, reps, hidden_state_size + predict_size * repetition))
    template[:, :, :, 0:hidden_state_size] = hidden_state
    read_start = 0
    write_start = hidden_state_size
    path = logs_dir + "/out_midi/"

    for i in range(repetition):
        feed_dict = {model.input_music_seg : template[:, :, :, read_start:read_start + hidden_state_size]}
        predict = session.run(model.predict, feed_dict=feed_dict)
                
        predict[predict < 0.9] = 0
        predict[predict >= 0.9] = 1        
        
        write_end = write_start + predict_size
        template[:, :, :, write_start:write_end] = predict
        write_start += predict_size
        read_start += predict_size

    template = _3d_tensor_to_2d_tensor(template)
    for i in range(batch_size):
        tensor2midi.save_tensor_to_midi(template[i], path + "TEST_MUSIC_" + str(i) + "_" + str(idx), tick_interval)
    print("****************************************************** ")
    print("                   TEST MUSIC SAVED                    ")
    print("****************************************************** ")        


def _3d_tensor_to_2d_tensor(tensor):
    batch_size, octav, segs, state_size = tensor.shape
    reshaped_tensor = np.zeros([batch_size, octav * segs, state_size, 1], dtype=np.uint8)
    
    for i in range(state_size):
        timeseg = np.zeros([batch_size, octav*segs])
        for j in range(segs):
            timeseg[:, j*octav:(j+1)*octav] = tensor[:, :, j, i]
        reshaped_tensor[:, :, i, 0] = timeseg
    return reshaped_tensor


def save_music(hidden_state, predict, path, name, batch_size, tick_interval):
    merged = np.concatenate((hidden_state, predict), axis=2)

    for i in range(batch_size):
        tensor2midi.save_tensor_to_midi(merged[i], path + name + "_" + str(i), tick_interval)
    print("*************** VALIDATION MUSIC SAVED *************** ")
    
    
