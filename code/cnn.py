import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import keras
import keras.layers as L
from keras.preprocessing import text, sequence
import pandas as pd
import numpy as np
from glovevectorizer import load_glove_weights, generate_weights

# BASE_DIR = '/home/kwu14/data/cs584_course_project'
BASE_DIR = '../data/'

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
    train_df['target'] = np.where(train_df['target'] >= 0.5, True, False)

    for column in IDENTITY_COLUMNS:
        train_df[column] = np.where(train_df[column] >= 0.5, True, False)

    sample_weights = np.ones(train_df.shape[0], dtype=np.float32)
    sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
    sample_weights += train_df['target'] * \
        (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
    sample_weights += (~train_df['target']) * \
        train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
    sample_weights /= sample_weights.mean()

    text_train = train_df['comment_text'].astype(str).values
    y_train = train_df['target'].values
    y_aux_train = train_df[AUX_COLUMNS].values

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

    return seq_train, y_train, y_aux_train, \
        test_df.id, x_test, weights, sample_weights


def load_cnn_model(weights):
    sequence_input = L.Input(shape=(MAX_LEN,))
    embedded_sequences = L.Embedding(
        weights.shape[0], weights.shape[1],
        weights=[weights], input_length=MAX_LEN,
        trainable=False)(sequence_input)

    l_cov1 = L.Conv1D(128, 2, activation='relu')(embedded_sequences)
    l_pool1 = L.MaxPooling1D(5)(l_cov1)
    l_cov2 = L.Conv1D(128, 3, activation='relu')(l_pool1)
    l_pool2 = L.MaxPooling1D(5)(l_cov2)
    l_cov3 = L.Conv1D(128, 4, activation='relu')(l_pool2)
    l_pool3 = L.MaxPooling1D(3)(l_cov3)  # global max pooling
    l_flat = L.Flatten()(l_pool3)
    out = L.Dense(1, activation='sigmoid')(l_flat)
    # aux_out = L.Dense(1, activation='sigmoid')(l_flat)
    model = keras.models.Model(inputs=sequence_input, outputs=out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
    )
    model.summary()
    return model


def load_model(weights, num_filters=3):
    words = L.Input(shape=(None, ))
    x = L.Embedding(weights.shape[0], weights.shape[1], weights=[weights],
                    input_length=MAX_LEN, trainable=False)(words)

    conv1 = L.Conv1D(num_filters, 2)(x)
    conv2 = L.Conv1D(num_filters, 3)(x)
    conv3 = L.Conv1D(num_filters, 4)(x)
    pool1 = L.MaxPool1D(pool_size=MAX_LEN-1)(conv1)
    pool2 = L.MaxPool1D(pool_size=MAX_LEN-2)(conv2)
    pool3 = L.MaxPool1D(pool_size=MAX_LEN-3)(conv3)

    concat = L.Concatenate(axis=1)([pool1, pool2, pool3])
    flat = L.Flatten()(concat)

    out = L.Dense(64, activation='relu')(flat)
    out = L.Dense(1, activation='sigmoid')(out)

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
    x_train, y_train, y_aux_train, \
        test_id, x_test, weights, sample_weights = load_data()

    checkpoint = keras.callbacks.ModelCheckpoint(
        'cnn.model.h5', save_best_only=True, verbose=1)
    es = keras.callbacks.EarlyStopping(patience=3, verbose=1)
    model = load_model(weights, num_filters)
    # model = load_cnn_model(weights)
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[es, checkpoint],
        verbose=2,
        # sample_weight=sample_weights.values
    )

    # evaluation
    model.load_weights('cnn.model.h5')
    test_preds = model.predict(x_test)
    # print(test_preds.flatten())
    submission = pd.DataFrame.from_dict({
        'id': test_id,
        'prediction': test_preds.flatten()
    })
    submission.to_csv('cnn_bias_submission.csv', index=False)
