import tensorflow as tf
import numpy as np


class TESSTER(object):
    def __init__(self, batch_size, is_training, num_keys, input_length, output_length, learning_rate):
        pass
    
    def conversation(self, input_tensor, session):
        retval = np.zeros_like(input_tensor)
        for i in range(len(retval)):
            idx = np.random.random(len(retval[i])) * 0.66
            retval[i] = np.round(idx)
        return retval
        
        
def main():
    sess = tf.Session()
    input_tensor=np.zeros((128, 256))
    tester = TEASTER(10, True, 128, 256, 256, 0.0001)
    tester.conversation(input_tensor)



