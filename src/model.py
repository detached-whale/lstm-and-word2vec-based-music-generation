from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm


# TODO : change to pytorch and add hyperparameter args
def note_prediction_model(vocab_size, embedding_dim, max_len, pretrained_vec, note_len):
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size,
                             output_dim=embedding_dim,
                             input_length=max_len,
                             weights=[pretrained_vec]))
    model.add(LSTM(units=embedding_dim,
                        return_sequences=True,
                        recurrent_dropout=0.3,))
    model.add(LSTM(units=embedding_dim,
                        recurrent_dropout=0.3,))
    model.add(BatchNorm())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))

    model.add(Dense(note_len))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def duration_prediction_model(vocab_size, embedding_dim, max_len, pretrained_vec, duration_len):
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                input_length=max_len,
                                weights=[pretrained_vec]))

    model.add(LSTM(units=embedding_dim,
                            return_sequences=True,
                            recurrent_dropout=0.3,))
    model.add(LSTM(units=embedding_dim,
                            recurrent_dropout=0.3,))
    model.add(BatchNorm())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))

    model.add(Dense(duration_len))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
