import music21
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import Dense, Dot, Embedding, Input, Reshape
from tensorflow.keras.models import Model
from itertools import chain
import numpy as np


def preporcess(midies, t_key = 'C', embedding_dim = 300, time_step = 20):
    notes, durations = parse_midi_file(midies, t_key)

    unique_duration = set(sum(durations, [] ))
    unique_duration_len = len(unique_duration)
    duration_to_int = dict((note, number) for number, note in enumerate(unique_duration))
    int_to_duration = dict((number, note) for number, note in enumerate(unique_duration))
    
    unique_note = set(sum(notes, []))
    unique_note_len = len(unique_note)
    note_to_int = dict((note, number) for number, note in enumerate(unique_note))
    int_to_note = dict((number, note) for number, note in enumerate(unique_note))

    unique_all = set(chain(unique_duration, unique_note))
    all_to_int = dict((note, number) for number, note in enumerate(unique_all))
    int_to_all = dict((number, note) for number, note in enumerate(unique_all))

    vocab_size = len(unique_all)

    indexed_notes = []

    for i in range(len(notes)):
        temp = []

        for j in range(len(notes[i])):
            temp.append(all_to_int[notes[i][j]])
            temp.append(all_to_int[notes[i][j]])

        indexed_notes.append(temp)

    X, Y = generating_wordpairs(indexed_notes, vocab_size, time_step)

    Y = np.array(Y)

    word_target, word_context = zip(*X)

    word_target = np.array(word_target, dtype=np.int32)
    word_context = np.array(word_context, dtype=np.int32)

    input_target = Input((1,))
    input_context = Input((1,))

    embedding_layer = Embedding(vocab_size, embedding_dim, input_length=1)

    target_embedding = embedding_layer(input_target)
    target_embedding = Reshape((embedding_dim, 1))(target_embedding)

    context_embedding = embedding_layer(input_context)
    context_embedding = Reshape((embedding_dim, 1))(context_embedding)

    hidden_layer = Dot(axes=1)([target_embedding, context_embedding])
    hidden_layer = Reshape((1,))(hidden_layer)

    output = Dense(16, activation='sigmoid')(hidden_layer)
    output = Dense(1, activation='sigmoid')(output)

    model_word2vec = Model(inputs=[input_target, input_context], outputs=output)
    model_word2vec.compile(loss='binary_crossentropy', optimizer='sgd')
    #model_word2vec.fit([word_target, word_context], Y, batch_size=64, epochs=10, verbose=1, shuffle=True)
    model_word2vec.fit([word_target, word_context], Y, batch_size=64, epochs=1, verbose=1, shuffle=True)

    pretrained_vector = model_word2vec.get_weights()[0]


    return notes, unique_note_len, durations, unique_duration_len, note_to_int, int_to_note, duration_to_int, int_to_duration, all_to_int, vocab_size, pretrained_vector


def generating_wordpairs(indexed_notes, vocab_size, window_size=20):
    X = []
    Y = []

    for notes in indexed_notes:
        x, y = skipgrams(sequence=notes, vocabulary_size=vocab_size, window_size=window_size,
        negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)

        X = X + list(x)
        Y = Y + list(y)

    return X, Y


def parse_midi_file(midies, t_key):
    all_notes = []
    all_durations = []

    for midi in midies:
        notes = []
        durations = []

        key = midi.analyze('key')
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch(t_key))
        transposed = midi.transpose(interval)

        try: # file has instrument parts
            temp = music21.instrument.partitionByInstrument(transposed)
            notes_to_parse = temp.parts[0].recurse() 

        except: # file has notes in a flat structure
            notes_to_parse = transposed.flat.notes

        prev_offset = -1

        for element in notes_to_parse:
            if isinstance(element, music21.note.Note):
                if element.offset == prev_offset:
                    notes[-1] = notes[-1] + ' ' + str(element.pitch)
                else:
                    notes.append(str(element.pitch))
                    durations.append(str(element.duration.quarterLength))
                prev_offset = element.offset

            elif isinstance(element, music21.chord.Chord):
                temp_note = ''
                for ePitch in element.pitches:
                    temp_note = temp_note + ' ' + str(ePitch)

                durations.append(str(element.duration.quarterLength))
                notes.append(temp_note.lstrip())

        if len(notes) > 100:
            all_notes.append(notes)
            all_durations.append(durations)

    return all_notes, all_durations
