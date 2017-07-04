import os
import numpy as np
import mido
from PIL import Image

is_message_print = False

class Dataset:
    def __init__(self, directory, batch_size, hidden_state_size, predict_size, num_keys, step=1):
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
        self.path = directory

        self.num_keys = num_keys

        self._read_midi_path(directory)
        self.file_size = len(self.files)
        self._read_next_file()

    def _read_midi_path(self, directory):
        subdirect = os.listdir(directory)
        for i, subdirect in enumerate(subdirect):
            if "." in subdirect:
                continue
            files = os.listdir(directory + "/"+subdirect + "/")
            for file in files:
                if ".mid" in file:
                    self.files.append(directory + "/" + subdirect + "/" + file)

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
        notes_input = np.zeros([self.batch_size, self.hidden_state_size, 128], dtype=np.float32)
        ground_truth = np.zeros([self.batch_size, self.predict_size, self.num_keys], dtype=np.float32)
        for i in range(self.batch_size):
            if self._calc_next_batch_offset():
                self._read_next_file()
            idx_from = self.batch_offset
            idx_to = idx_from +self.hidden_state_size

            input_segment = self.current_midi[idx_from:idx_to, : ]
            gt_segment = self.current_midi[idx_to:idx_to + self.predict_size, :]
            notes_input[i, 0:len(input_segment)] = input_segment
            ground_truth[i] = gt_segment
        return notes_input, ground_truth

    def _read_next_file(self):
        current_midi = np.array((0))
        while np.sum(current_midi) == 0:
            filename = self.files[self.file_offset]
            # try:
            current_midi = midi2tensor(filename, self.num_keys)
            # except:
            #     current_midi = np.zeros((1))
            if np.sum(current_midi) == 0:
                os.popen("rm " + self.files[self.file_offset])
                print(self.files[self.file_offset] + "is not appropriate midi file")
                self.files.remove(self.files[self.file_offset])
                self.file_size -= 1
            self.file_offset += 1
            if self.file_offset >= self.file_size-1:
                self._read_midi_path(self.path)
                self.file_offset = 0

        self.current_midi = current_midi


def main():
    train_dataset_reader = Dataset("train_data/", 1100, 1500, 200, 128)
    # while True:
    #     train_dataset_reader.next_batch()

def midi2tensor(path, num_keys):
    mid = mido.MidiFile(path)

    time_duration = mid.length
    num_segments = int(time_duration / 0.06)
    ticks_per_beat = mid.ticks_per_beat
    if is_message_print:
        print(mid.ticks_per_beat)
    tensor = np.zeros((num_segments, num_keys), dtype=np.float32)
    meta = []
    tracks = []
    if is_message_print:
        print('opening ' + path)
    for track in mid.tracks:
        if is_message_print:
            print('track')
        for msg in track:
            if is_message_print:
                print(msg)

    for i, track in enumerate(mid.tracks):
        if is_message_print:
            print(track)
            print('Track {}: {}'.format(i, track.name))
        if i == 0:
            meta.append(track)
        else:
            tracks.append(track)

    # for i in range(len(mid.tracks)):
    #     track_ = mid.tracks[i]
    #     print(track_)
    #     if track_[0].is_meta:
    #         meta.append(track_.clone())
    #     else:
    #         tracks.append(track_.clone())

    meta_tempo = parse_meta(meta[0], num_segments, ticks_per_beat)

    for track in tracks:
        if is_message_print:
            print('well', track)
        tensor += parse_track(track, num_segments, meta_tempo, num_keys)
    tensor[tensor>0] = 1
    # for line in tensor:
    #     print(line)
#    rgb_tensor = np.array((tensor))
    result = Image.fromarray(np.uint8(tensor*255))
    result.save("test.png")
    return tensor


def parse_meta(meta, num_segments, ticks_per_beat):
    meta_tempo = np.zeros((num_segments), dtype=np.float32)
    start = 0
    end = 0
    previous_tempo = 0
    for msg in meta:
        splits = str(msg).split(" ")
        time = int(splits[-1][5:-1])
        if previous_tempo > 0:
            end = start + int(mido.tick2second(time, ticks_per_beat, previous_tempo)/0.06)
            meta_tempo[start:end] = int(mido.second2tick(0.06, ticks_per_beat, previous_tempo))
        if "set_tempo" in str(msg):
            previous_tempo = int(splits[3][6:])
        start = end
    return meta_tempo


def parse_track(track, num_segments, meta_tempo, num_keys):
    ret_tensor = np.zeros((num_segments, num_keys), dtype=np.float32)
    prev_on_notes = np.zeros(num_keys, dtype=np.float32)
    count = 0
    for msg in track:
        if "time" not in str(msg):
            continue

        splits = str(msg).split(" ")
        time = splits[-1][5:]
        if time[-1] == '>':
            time = time[:-1]
        time = int(time)

        if time > 0:
            while time > 0 and count < num_segments:
                time -= int(meta_tempo[count])
                ret_tensor[count, :] = prev_on_notes
                count += 1
            time = -1

        if splits[0] == "note_on":
            note = int(splits[2][5:]) - 1
            velocity = int(splits[3][9:])
            if velocity > 0:
                prev_on_notes[note] += velocity
            else:
                prev_on_notes[note] = 0

        if time > 0:
            while time > 0 and count < num_segments:
                time -= int(meta_tempo[count])
                ret_tensor[count, :] = prev_on_notes
                count += 1
            time = -1

        if splits[0] == "note_off":
            note = int(splits[2][5:]) - 1
            prev_on_notes[note] = 0

        if "note" not in splits[0]:
            prev_on_notes *= 0

        if time == 0 and count < num_segments:
            ret_tensor[count, :] = prev_on_notes

    return ret_tensor


def qkey2int(key, num_keys):
    b, l = key.shape
    retval = np.zeros((b), dtype=np.int)

    for batch in range(b):
        for i in range(num_keys):
            if key[batch, i] != 0:
                retval[batch] += 2**(127-i)
    return retval



def key2int(key):
    b, l = key.shape
    retval = np.zeros((b), dtype=np.int)

    for batch in range(b):
        retval[batch] = np.argmax(key[batch])
    return retval

if __name__== "__main__":
    main();