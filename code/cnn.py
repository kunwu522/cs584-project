import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import keras
from keras.preprocessing import text, sequence
import pandas as pd
import numpy as np

from keras import backend as K
from glovevectorizer import load_glove_weights, generate_weights

BASE_DIR = '/home/kwu14/data/cs584_course_project'
# BASE_DIR = '../data/'

DATA_SIZE = 100000
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

    text_train = train_df['comment_text'].astype(str).values
    y_train = train_df['target'].values

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

    return seq_train, y_train, x_test, weights

def load_cnn_model(weights):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = keras.layers.Embedding(weights.shape[0], weights.shape[1],
                               weights=[weights],
                               input_length=MAX_LEN,
                               trainable=False)(sequence_input)
    l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(1, activation='sigmoid')(l_dense)
    model = keras.models.Model(inputs=words, output=out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
    )
    model.summary()
    return model


def load_model(weights, num_filters=3):
    words = keras.layers.Input(shape=(None, ))
    x = keras.layers.Embedding(weights.shape[0], weights.shape[1],
                               weights=[weights],
                               input_length=MAX_LEN,
                               trainable=False)(words)

    conv1 = keras.layers.Conv1D(num_filters, 2)(x)
    conv2 = keras.layers.Conv1D(num_filters, 3)(x)
    conv3 = keras.layers.Conv1D(num_filters, 4)(x)
    pool1 = keras.layers.MaxPool1D(pool_size=MAX_LEN-1)(conv1)
    pool2 = keras.layers.MaxPool1D(pool_size=MAX_LEN-2)(conv2)
    pool3 = keras.layers.MaxPool1D(pool_size=MAX_LEN-3)(conv3)

    concat = keras.layers.Concatenate(axis=1)([pool1, pool2, pool3])
    flat = keras.layers.Flatten()(concat)

    out = keras.layers.Dense(64, activation='relu')(flat)
    out = keras.layers.Dense(1, activation='sigmoid')(out)

    model = keras.models.Model(inputs=words, output=out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
    )
    model.summary()
    return model


if __name__ == "__main__":
    # hyper-paramters
    batch_size = 1024
    epochs = 50
    num_filters = 5

    # load data
    x_train, y_train, x_test, weights = load_data()

    checkpoint = keras.callbacks.ModelCheckpoint(
        'cnn.model.h5', save_best_only=True, verbose=1)
    es = keras.callbacks.EarlyStopping(patience=3, verbose=1)
    # model = load_model(weights, num_filters)
    model = load_cnn_model(weights)
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[es, checkpoint],
        verbose=2
    )

    # evaluation
    model.load_weights('cnn.model.h5')
    test_preds = model.predict(x_test)

    submission = pd.read_csv('./sample_submission.csv', index_col='id')
    submission['prediction'] = test_preds
    submission.reset_index(drop=False, inplace=True)
    submission.head()
    submission.to_csv('../outputs/cnn_submission.csv', index=False)
