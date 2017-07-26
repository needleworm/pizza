import os

import mido
import numpy as np

import tensor2midi

is_message_print = False
is_test = False


class Dataset:
    """
    traning data를 생성하기 위한 Dataset 클래스.
    """

    def __init__(self, directory, batch_size, is_batch_zero_pad, hidden_state_size, predict_size, num_keys,
                 tick_interval, step=200):
        """
        :param directory: directory of directories storing midi files
        :param batch_size: batch size
        :param is_batch_zero_pad: batch가 zero padding 방식인지 (True/False)
        :param hidden_state_size: hidden state의 size:
        :param predict_size: predict할 ground truth의 size:
        :param num_keys: key의 길이 (default: 128):
        :param tick_interval: midi를 커팅할 interval의 크기:
        :param step: step size per batch call
        """
        print("Initializing Midi File Reader...")

        self.files = []
        self.step = step
        self.current_midi = None
        self.current_midi_size = 0

        self.tensor_list = []

        self.batch_offset = 0
        self.batch_size = batch_size
        self.file_offset = 0
        self.predict_size = predict_size
        self.hidden_state_size = hidden_state_size
        self.path = directory
        self.tick_interval = tick_interval

        self.is_batch_zero_pad = is_batch_zero_pad

        self.num_keys = num_keys

        self._read_midi_path(directory)
        self.file_size = len(self.files)
        self._read_next_file()

    def _read_midi_path(self, directory):
        """
        directory의 subdirectory 안에서 midi의 경로를 읽어 self.files에 경로를 저장한다.
        :param directory: 찾아볼 directory
        :return: None
        """
        subdirect = os.listdir(directory)
        for i, subdirect in enumerate(subdirect):
            if "." in subdirect:
                continue
            files = os.listdir(directory + "/" + subdirect + "/")
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
            return True
        return False

    def next_batch(self):
        """
        Next batch를 set up한다.
        주어진 midi 파일 내에서 batch를 set up 하지 못할 경우 다음 midi 파일을 읽어 진행한다.
        :return: notes_input (hidden state의 집합), ground_truth (predict의 검증값)
        """
        if self.is_batch_zero_pad:
            notes_input = np.zeros([self.batch_size, self.num_keys, self.hidden_state_size, 1], dtype=np.float32)
            ground_truth = np.zeros([self.batch_size, self.num_keys, self.predict_size, 1], dtype=np.float32)
            for i in range(self.batch_size):
                self.batch_offset += self.step
                if self._zero_pad(self.current_midi, notes_input[i], ground_truth[i], self.batch_offset,
                                  self.hidden_state_size, self.predict_size):
                    pass
                else:
                    while not self._zero_pad(self.current_midi, notes_input[i], ground_truth[i], self.batch_offset,
                                             self.hidden_state_size, self.predict_size):
                        self._read_next_file()
            return notes_input, ground_truth
        else:
            notes_input = np.zeros([self.batch_size, self.num_keys, self.hidden_state_size, 1], dtype=np.float32)
            ground_truth = np.zeros([self.batch_size, self.num_keys, self.predict_size, 1], dtype=np.float32)
            for i in range(self.batch_size):
                if self._calc_next_batch_offset():
                    self._read_next_file()
                idx_from = self.batch_offset
                idx_to = idx_from + self.hidden_state_size

                input_segment = self.current_midi[:, idx_from:idx_to]
                gt_segment = self.current_midi[:, idx_to:idx_to + self.predict_size]
                notes_input[i, :, 0:len(input_segment[0]), 0] = input_segment
                ground_truth[i, :, 0:len(gt_segment[0]), 0] = gt_segment
            return notes_input, ground_truth
            
    def next_batch_3d(self):
        notes, gts = self.next_batch()
        if self.num_keys % 8 != 0:
            print("num_keys must be times of 8!")
            exit(1)
            
        reshaped_notes = np.zeros([self.batch_size, 8, int(self.num_keys / 8), self.hidden_state_size], np.uint8)
        reshaped_gts = np.zeros([self.batch_size, 8, int(self.num_keys / 8), self.predict_size], np.uint8)
        
        for i in range(self.hidden_state_size):
            reshaped_timeseg = np.zeros([self.batch_size, 8, int(self.num_keys/8)])
            for j in range(int(self.num_keys/8)):
                reshaped_timeseg[:, :, j] = notes[:, j*8:(j+1)*8, i, 0]                
            reshaped_notes[:, :, :, i] = reshaped_timeseg
            
        for i in range(self.predict_size):
            reshaped_timeseg = np.zeros([self.batch_size, 8, int(self.num_keys/8)])
            for j in range(int(self.num_keys/8)):
                reshaped_timeseg[:, :, j] = notes[:, j*8:(j+1)*8, i, 0]                
            reshaped_gts[:, :, :, i] = reshaped_timeseg
            
        return reshaped_notes, reshaped_gts
        

    def _zero_pad(self, input, notes_output, gt_output, offset, notes_size, gt_size):
        """
        주어진 input을 zero padding해 output에 넣는다.
        :param input: 넣어주는 midi 값 (current_midi)
        :param notes_output: input을 처리해서 넣을 notes의 output [batch_size, num_keys. size, 1]의 배열
        :param gt_output: input을 처리해서 넣을 ground truth의 output [batch_size, num_keys. size, 1]의 배열
        :param offset: 현재 index의 offset값
        :param notes_size: note size
        :param gt_size: ground truth의 size
        :return: 성공하면 True, 실패하면 False
        """
        input_size = len(input[0])
        notes_input_start_index = offset - notes_size // 2
        notes_input_end_index = notes_input_start_index + notes_size
        notes_output_start_index = 0
        notes_output_end_index = notes_size
        if notes_input_start_index < 0:
            notes_output_start_index += -notes_input_start_index
            notes_input_start_index = 0

        gt_input_start_index = notes_input_end_index
        if gt_input_start_index + gt_size // 2 > input_size:
            return False
        gt_input_end_index = gt_input_start_index + gt_size
        gt_output_start_index = 0
        gt_output_end_index = gt_size
        if gt_input_end_index > input_size:
            # 주어진 ground truth의 마지막 index가 input size보다 크다면, 그만큼 사이즈를 줄여서, zero padding이 되도록한다.
            gt_output_end_index -= gt_input_end_index - input_size
            gt_input_end_index = input_size

        notes_output[:, notes_output_start_index:notes_output_end_index, 0] = input[:,
                                                                              notes_input_start_index:notes_input_end_index]
        gt_output[:, gt_output_start_index:gt_output_end_index, 0] = input[:, gt_input_start_index:gt_input_end_index]
        return True

    def _read_next_file(self):
        """
        다음 midi 파일을 읽어 tensor로 만든다.
        이때, 한 번 읽은 파일은 self.tensor_list에 (filename, tensor)의 형태로 저장해, 다음 번 로드시 불필요한 작업을
        거치지 않도록 했다.
        :return: None
        """
        if is_test:
            print('reading ' + self.files[self.file_offset])
        current_midi = np.array((0))
        while np.sum(current_midi) == 0:
            filename = self.files[self.file_offset]
            # try:
            current_midi = None
            for item in self.tensor_list:
                if item[0] == filename:
                    current_midi = item[1]
            if current_midi is None:
                current_midi = midi2tensor(filename, self.num_keys, self.tick_interval)
                self.tensor_list.append([filename, current_midi])
                if np.sum(current_midi) == 0:
                    os.popen("rm " + self.files[self.file_offset])
                    print(self.files[self.file_offset] + "is not appropriate midi file")
                    self.files.remove(self.files[self.file_offset])
                    self.file_size -= 1
            self.file_offset += 1
            if self.file_offset >= self.file_size - 1:
                self._read_midi_path(self.path)
                self.file_offset = 0

        self.current_midi = current_midi
        self.current_midi_size = len(current_midi[0])
        self.batch_offset = 0


def main():
    if is_test:
        train_dataset_reader = Dataset("train_data2/", 50000, 300, 300, 128, 0.03)
        while True:
            train_dataset_reader.next_batch()


def midi2tensor(path, num_keys, tick_interval):
    """
    주어진 path의 midi 파일을 (num_keys x (time_duration / tick_interval))의 크기를 가지는 tensor로 변환한다.
    :param path: midi 파일의 경로
    :param num_keys: key의 길이 (default: 128)
    :param tick_interval: midi를 커팅할 interval의 크기
    :return: tensor
    """
    #    print('opening ' + path)
    mid = mido.MidiFile(path)

    time_duration = mid.length
    num_segments = int(time_duration / tick_interval)
    ticks_per_beat = mid.ticks_per_beat
    if is_message_print:
        print(mid.ticks_per_beat)
    tensor = np.zeros((num_segments, num_keys), dtype=np.float32)
    meta = []
    tracks = []
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

    meta_tempo = parse_meta(meta[0], num_segments, ticks_per_beat, tick_interval)

    for track in tracks:
        if is_message_print:
            print('well', track)
        tensor += parse_track(track, num_segments, meta_tempo, num_keys)
    tensor[tensor > 0] = 1
    # for line in tensor:
    #     print(line)
    #    rgb_tensor = np.array((tensor))
    if is_test:
        # result = Image.fromarray(np.uint8(tensor*255))
        # result.save("test.png")
        tensor2midi.save_tensor_to_midi(tensor.transpose(), 'out_' + path.split('/')[-1], tick_interval)
    return tensor.transpose()


def parse_meta(meta, num_segments, ticks_per_beat, tick_interval):
    meta_tempo = np.zeros((num_segments), dtype=np.float32)
    start = 0
    end = 0
    previous_tempo = 0
    for msg in meta:
        splits = str(msg).split(" ")
        time = int(splits[-1][5:-1])
        if previous_tempo > 0:
            end = start + int(mido.tick2second(time, ticks_per_beat, previous_tempo) / tick_interval)
            meta_tempo[start:end] = int(mido.second2tick(tick_interval, ticks_per_beat, previous_tempo))
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
                retval[batch] += 2 ** (127 - i)
    return retval


def key2int(key):
    b, l = key.shape
    retval = np.zeros((b), dtype=np.int)

    for batch in range(b):
        retval[batch] = np.argmax(key[batch])
    return retval


if __name__ == "__main__":
    main()
