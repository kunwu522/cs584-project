import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from glovevectorizer import load_glove_weights, generate_weights
import pandas as pd
import numpy as np
import keras
from keras.preprocessing import text, sequence
from keras.layers import add, concatenate, Dense
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D


# BASE_DIR = '/home/kwu14/data/cs584_course_project'
BASE_DIR = '../data/'
VOCAB_SIZE = 10000
MAX_LEN = 166

AUX_COLUMNS = ['severe_toxicity', 'obscene',
               'identity_attack', 'insult', 'threat']
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


def load_data():
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_train.csv'))
    train_df['target'] = np.where(train_df['target'] >= 0.5, True, False)

    text_train = train_df['comment_text'].astype(str).values
    y_train = train_df['target'].values

    for column in IDENTITY_COLUMNS:
        train_df[column] = np.where(train_df[column] >= 0.5, True, False)

    sample_weights = np.ones(train_df.shape[0], dtype=np.float32)
    sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
    sample_weights += train_df['target'] * \
        (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
    sample_weights += (~train_df['target']) * \
        train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
    sample_weights /= sample_weights.mean()

    tk = text.Tokenizer(num_words=VOCAB_SIZE)
    tk.fit_on_texts(text_train)
    weights = generate_weights(
        load_glove_weights(os.path.join(BASE_DIR, 'glove.6B.300d.txt')),
        tk.word_index,
        VOCAB_SIZE - 1
    )

    seq_train = tk.texts_to_sequences(text_train)
    seq_train = sequence.pad_sequences(seq_train, maxlen=MAX_LEN)

    # load test data
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_test.csv'))
    x_test = test_df['comment_text'].astype(str).values
    x_test = tk.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    return seq_train, y_train, test_df.id, x_test, weights, sample_weights


def load_model(weights, hidden_size=100):
    words = keras.layers.Input(shape=(MAX_LEN, ), name='input_layer')
    x = keras.layers.Embedding(weights.shape[0], weights.shape[1],
                               weights=[weights], trainable=False,
                               name='embedding_layer')(words)
    x = keras.layers.Bidirectional(keras.layers.LSTM(hidden_size,
                                                     return_sequences=True,
                                                     name='lstm1'))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(hidden_size,
                                                     return_sequences=True,
                                                     name='lstm2'))(x)
    hidden = concatenate(
        [GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x), ])
    hidden = add([hidden, Dense(400, activation='relu')(hidden)])
    hidden = add([hidden, Dense(400, activation='relu')(hidden)])
    out = Dense(1, activation='sigmoid')(hidden)

    model = keras.models.Model(inputs=words, output=out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model


if __name__ == "__main__":
    # hyper-paramters
    batch_size = 1024
    epochs = 50
    hidden_size = 100

    # load data
    x_train, y_train, test_id, x_test, weights, sample_weights = load_data()

    checkpoint = keras.callbacks.ModelCheckpoint(
        'rnn_bias.model.h5', save_best_only=True, verbose=1)
    es = keras.callbacks.EarlyStopping(patience=3, verbose=1)
    model = load_model(weights, hidden_size)
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[es, checkpoint],
        verbose=2,
        sample_weight=sample_weights.values
    )

    # evaluation
    model.load_weights('rnn_bias.model.h5')
    test_preds = model.predict(x_test)

    submission = pd.DataFrame.from_dict({
        'id': test_id,
        'prediction': test_preds
    })
    submission.to_csv('rnn_bias_submission.csv', index=False)
