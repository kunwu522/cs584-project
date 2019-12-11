from glovevectorizer import load_glove_weights, generate_weights
import pandas as pd
from keras.preprocessing import text, sequence
import keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


BASE_DIR = '/home/kwu14/data/cs584_course_project'
# BASE_DIR = '../data/'
VOCAB_SIZE = 10000
MAX_LEN = 305


def load_data():
    train_pd = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_train.csv'))
    text_train = train_pd['comment_text'].astype(str).values
    y_train = train_pd['target'].values

    tk = text.Tokenizer(num_words=VOCAB_SIZE)
    tk.fit_on_texts(text_train)
    weights = generate_weights(
        load_glove_weights(os.path.join(BASE_DIR, 'glove.6B.300d.txt')),
        tk.word_index,
        VOCAB_SIZE
    )

    seq_train = tk.texts_to_sequences(text_train)
    seq_train = sequence.pad_sequences(seq_train, maxlen=MAX_LEN)

    # load test data
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_test.csv'))
    x_test = test_df['comment_text'].astype(str).values
    x_test = tk.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    return seq_train, y_train, test_df.id, x_test, weights


def load_model(weights, hidden_size=100):
    words = keras.layers.Input(shape=(None, ), name='input_layer')
    x = keras.layers.Embedding(weights.shape[0], weights.shape[1],
                               weights=[weights], trainable=False,
                               name='embedding_layer')(words)
    x = keras.layers.Bidirectional(keras.layers.LSTM(hidden_size,
                                                     return_sequences=False,
                                                     name='lstm1'))(x)

    x = keras.layers.Dense(100, activation='relu')(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)

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
    x_train, y_train, test_id, x_test, weights = load_data()

    checkpoint = keras.callbacks.ModelCheckpoint(
        'rnn.model.h5', save_best_only=True, verbose=1)
    es = keras.callbacks.EarlyStopping(patience=3, verbose=1)
    model = load_model(weights, hidden_size)
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[es, checkpoint],
        verbose=2
    )

    # evaluation
    model.load_weights('rnn.model.h5')
    test_preds = model.predict(x_test)

    submission = pd.DataFrame.from_dict({
        'id': test_id,
        'prediction': test_preds
    })
    submission.to_csv('rnn_submission.csv', index=False)
