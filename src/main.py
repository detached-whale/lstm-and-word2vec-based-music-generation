from data_loader import load_midi_files
from preprocessor import preporcess
from model import note_prediction_model, duration_prediction_model
import numpy as np
import copy
from keras.utils import np_utils
import music21
import argparse

parser = argparse.ArgumentParser()
# TODO : read model related arguments
args = parser.parse_args()

def main():
    midies, file_names = load_midi_files("../data")

    t_key = 'C'
    embedding_dim = 300
    time_step = 20

    notes, unique_note_len, durations, unique_duration_len, note_to_int, int_to_note, duration_to_int, int_to_duration, all_to_int, vocab_size, pretrained_vector  = preporcess(midies, t_key, embedding_dim, time_step)
    train_x = []
    train_note_y = []
    train_duration_y = []

    for i in range(len(notes)):
        temp_x = []
        temp_note_y = []
        temp_duration_y = []

        for j in range(len(notes[i]) - time_step):
        temp_x_2 = []
        for k in range(time_step):
            temp_x_2.append(all_to_int[notes[i][j+k]])
            temp_x_2.append(all_to_int[durations[i][j+k]])
        temp_x.append(temp_x_2)
        temp_note_y.append(note_to_int[notes[i][j+time_step]])
        temp_duration_y.append(duration_to_int[durations[i][j+time_step]])

        train_x.append(temp_x)
        train_note_y.append(temp_note_y)
        train_duration_y.append(temp_duration_y)

    final_train_x = sum(train_x, [])

    max_len = 0
    for tx in final_train_x:
        if max_len < len(tx):
            max_len = len(tx)

    final_train_x = np.array(final_train_x)

    np_train_note_y = copy.deepcopy(train_note_y)
    np_train_duration_y = copy.deepcopy(train_duration_y)

    np_train_note_y = sum(np_train_note_y, [])
    np_train_note_y = np.array(np_train_note_y)
    np_train_note_y = np_utils.to_categorical(np_train_note_y)

    np_train_duration_y = sum(np_train_duration_y, [])
    np_train_duration_y = np.array(np_train_duration_y)
    np_train_duration_y = np_utils.to_categorical(np_train_duration_y)

    indices = np.random.permutation(final_train_x.shape[0])
    train_indices = indices[:int(len(indices) * 0.8)]
    train_doc = final_train_x[train_indices,:]
    train_note_y = np_train_note_y[train_indices,:]
    train_dur_y = np_train_duration_y[train_indices,:]

    model_for_note = note_prediction_model(vocab_size, embedding_dim, max_len, pretrained_vector, unique_note_len)
    model_for_note.fit(train_doc, train_note_y, batch_size=128, epochs=50, verbose=1, shuffle=True)

    model_for_duration = duration_prediction_model(vocab_size, embedding_dim, max_len, pretrained_vector, unique_duration_len)
    model_for_duration.fit(train_doc, train_dur_y, batch_size=128, epochs=50, verbose=1, shuffle=True)

    prev = 0
    for i in range(len(train_x)):
        test_x = copy.deepcopy(final_train_x[prev])
        test_x = np.reshape(test_x, (-1, len(test_x)))

        created_notes = []
        created_durs = []

        for j in range(len(train_x[i])):
            pred_note = model_for_note.predict(test_x)
            pred_dur = model_for_duration.predict(test_x)

            created_note = int_to_note[np.argmax(pred_note)]
            created_dur = int_to_note[np.argmax(pred_dur)]
            created_notes.append(created_note)
            created_durs.append(created_dur)

            for k in range(19, 0, -1):
                test_x[0][k*2] = test_x[0][(k-1)*2]
                test_x[0][k*2 + 1] = test_x[0][(k-1)*2]

            test_x[0][0] = all_to_int[created_note]
            test_x[0][1] = all_to_int[created_dur]

        # TODO : decouple postprocess part
        out = music21.stream.Stream()

        for idx in range(len(created_notes)):
            if len(created_notes[idx].split(' ')) != 1:
                new_note = music21.chord.Chord(created_notes[idx])
            else:
                new_note = music21.note.Note(created_notes[idx])

            cDur = music21.duration.Duration()
            try:
                cDur.quarterLength = float(created_durs[idx])
            except:
                cDur.quarterLength = 1

            new_note.duration = cDur
            out.append(new_note)

        out.write('midi', fp='../output/output-' + str(i) + '.mid')

        prev += len(train_x[i])

if __name__ == '__main__':
    main()
