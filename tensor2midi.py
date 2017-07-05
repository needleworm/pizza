from datetime import datetime
import os
from time import gmtime, strftime

from music21 import *
import numpy as np

import midi2tensor

bpm = 120


def save_tensor_to_midi(tensor, filename, tick_interval):
    """
    tensor를 midi 파일로 저장.
    :param tensor: 주어진 tensor를 의미
    :param filename: filename.
    :return: None
    """
    # print(len(tensor))
    s = stream.Stream()
    s.insert([0, tempo.MetronomeMark(number=bpm)])
    for pitch, line in enumerate(tensor):
        i = 0
        while i < len(line):
            offset = i
            if line[i]:
                note_len = 1
                while i < len(line) - 1 and line[i+1]:
                    note_len += 1
                    i += 1
                # print(offset, i, tick_to_quarterLength(offset, tick_interval), tick_to_quarterLength(note_len, tick_interval))
                s.insert(np.round(tick_to_quarterLength(offset, tick_interval), 3), note.Note(pitch, quarterLength=tick_to_quarterLength(note_len, tick_interval)))
            i += 1
    # save to midi
    # s.show('text', addEndTimes=True)
    owd = os.getcwd()
    os.chdir(owd)
    mf = midi.translate.streamToMidiFile(s)
    mf.open(filename + '.midi', 'wb')
    mf.write()
    mf.close()


def tick_to_quarterLength(tick, tick_interval):
    """
    tick을 quarterLength로 변환한다.
    :param tick: tick의 길이
    :return: quarterLength로 변환한 tick
    """
    return tick * tick_interval * bpm / 60
