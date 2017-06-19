import os
import numpy as np
from music21 import *

class Dataset:
    def __init__(self, directory, batch_size, hidden_state_size, predict_size, step=1):
        """
        :param directory: directory of directories storing midi files
        :param batch_size: batch size
        :param sliding: prediction size
        :param step: step size per batch call
        """
        print("Initializing Midi File Reader...")

        self.files = []
        self.step = step
        self.current_midi = None
        self.current_midi_size = 0

        self.batch_offset = 0
        self.batch_size = batch_size
        self.file_offset = 0
        self.predict_size = predict_size
        self.hidden_state_size = hidden_state_size

        self._read_midi_path(directory)
        self.file_size = len(self.files)
        self._read_next_file()

    def _read_midi_path(self, directory):
        for i, subdirect in enumerate(directory):
            files = os.listdir(subdirect + "/")
            for file in files:
                if ".mid" in file:
                    self.files.append(directory + "/" + file)

    def _calc_next_batch_offset(self):
        """
        return True if need to change music.
        return False if not need to changer music.
        """
        self.batch_offset += self.step
        if self.batch_offset + self.predict_size + self.hidden_state_size >= self.current_midi_size:
            self.batch_offset = 0
            return True
        return False

    def next_batch(self):
        notes_input = np.zeros([self.batch_size, 88, self.hidden_state_size], dtype=np.float32)
        ground_truth = np.zeros([self.batch_size, 88, self.predict_size], dtype=np.float32)
        for i in range(self.batch_size):
            idx_from = self.batch_offset
            idx_to = idx_from + self.hidden_state_size
            if self._calc_next_batch_offset():
                self._read_next_file()
            input_segment = self.current_midi[:, idx_from:idx_to]
            gt_segment = self.current_midi[:, idx_from + idx_to:idx_from + idx_to + self.predict_size]
            notes_input[i] = input_segment
            ground_truth[i] = gt_segment
        return notes_input, ground_truth

    def _read_next_file(self):
        current_midi = -1
        while current_midi == -1:
            file = self.files[self.file_offset]
            self.file_offset += 1
            if self.file_offset >= self.file_size:
                self.file_offset = 0
            current_midi = midi_reader(file)
        self.current_midi = current_midi


def midi_reader(file):
    """
    File을 읽어옴.
    n <-- 곡 빠르기와 상관 없이 0.06초 단위로 길이를 쪼갬.
    retval = np.zeros((88, n))
    for i, tracks in enumerate(midi):
        for j in range(n):
            for k in range(88):
                if k번 건반 in j th time step is pressed:
                    retval[k, n] = 1
    return retval
    """
    """
    try, except로 예외처리 할 경우, 에러 뜨면 -1 리턴.
    """
    return 0
