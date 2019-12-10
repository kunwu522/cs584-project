import os
import keras
from keras.preprocessing import text, sequence
import pandas as pd

from glovevectorizer import load_glove_weights, generate_weights

BASE_DIR = '/home/kwu14/data/cs584_course_project'
max_len = 0


def load_data():
    train_pd = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_train.csv'))
    text_train = train_pd['comment_text'].astype(str).values
    y_train = train_pd['target'].values

    tk = text.Tokenizer(num_words=10000)
    tk.fit_on_texts(text_train)

    weights = generate_weights(
        load_glove_weights(os.path.join(BASE_DIR, 'glove.6B.300d.txt')),
        tk.word_index
    )

    seq_train = tk.texts_to_sequences(text_train)

    max_len = 0
    for seq in seq_train:
        max_len = max(len(seq), max_len)

    seq_train = sequence.pad_sequences(seq_train, maxlen=max_len)

    # load test data
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_test.csv'))
    x_test = test_df['comment_text'].astype(str).values
    x_test = tk.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    return seq_train, y_train, x_test, weights


def load_model(weights):
    words = keras.layers.Input(shape=(max_len,))
    x = keras.layers.Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=False)(words)
    conv1 = keras.layers.Conv1D(10, 2)(x)
    conv2 = keras.layers.Conv1D(10, 3)(x)
    conv3 = keras.layers.Conv1D(10, 4)(x)
    pool1 = keras.layers.MaxPool1D(pool_size=conv1.shape[0])(conv1)
    pool2 = keras.layers.MaxPool1D(pool_size=conv2.shape[0])(conv2)
    pool3 = keras.layers.MaxPool1D(pool_size=conv3.shape[0])(conv3)
    concat = keras.layers.Concatenate()([pool1, pool2, pool3])
    out = keras.layers.Dense(1, activation='sigmoid')(concat)

    model = keras.models.Model(input=words, output=out)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['acc', 'f1'])


if __name__ == "__main__":
    # hyper-paramters
    batch_size = 1024
    epochs = 50

    # load data
    x_train, y_train, x_test, weights = load_data()

    checkpoint = keras.callbacks.ModelCheckpoint(
        'cnn.model.h5', save_best_only=True)
    es = keras.callbacks.EarlyStopping(patience=3)
    model = load_model(weights)
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[es, checkpoint]
    )

    # evaluation
    model.load_weights('my_model_weights.h5')
    test_preds = model.predict(x_test)

    submission = pd.read_csv('./sample_submission.csv', index_col='id')
    submission['prediction'] = test_preds
    submission.reset_index(drop=False, inplace=True)
    submission.head()
    submission.to_csv('../outputs/cnn_submission.csv', index=False)
