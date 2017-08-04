import tensorflow as tf
import numpy as np
import test as piano # piano module


"""
    model
"""
class TESTER(object):
    def __init__(self, batch_size, is_training, num_keys, input_length, output_length, learning_rate):
        pass
    
    def conversation(self, input_tensor, session):
        retval = np.zeros_like(input_tensor)
        for i in range(len(retval)):
            idx = np.random.random(len(retval[i])) * 0.66
            retval[i] = np.round(idx)
        return retval
        



sess = tf.Session()
tester = TESTER(10, True, 128, 256, 256, 0.0001)




piano = piano.Keyboard() # initial
while piano.run:   
    while not(piano.is_ai):
        piano.play_input()
    input_tensor = piano.get_motive()
    piano.play_ai(tester.conversation(input_tensor, sess)) # press -> button
    piano.is_ai = False