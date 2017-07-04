import os

from music21 import *
import numpy as np

bpm = 120

def save_tensor_to_midi(tensor):
    print(len(tensor))
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
                print(offset, i, tick_to_quarterLength(offset), tick_to_quarterLength(note_len))
                s.insert(np.round(tick_to_quarterLength(offset), 3), note.Note(pitch, quarterLength=tick_to_quarterLength(note_len)))
            i += 1
    # save to midi
    s.show('text', addEndTimes=True)
    owd = os.getcwd()
    os.chdir(owd)
    mf = midi.translate.streamToMidiFile(s)
    mf.open('out.midi', 'wb')
    mf.write()
    mf.close()

def tick_to_quarterLength(tick):
    return tick * 0.01 * bpm / 60
