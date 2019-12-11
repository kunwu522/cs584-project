import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
from keras.preprocessing import text
from keras.engine.topology import Layer
from keras.layers import Bidirectional, LSTM, TimeDistributed
from keras.layers import Input, Embedding, Dense, Dropout
from keras.models import Model
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize

from glovevectorizer import load_glove_weights, generate_weights

BASE_DIR = '/home/kwu14/data/cs584_course_project'
# BASE_DIR = '../data/'

VOCAB_SIZE = 10000

MAX_SENTS = 43
MAX_SENT_LEN = 1000

AUX_COLUMNS = ['severe_toxicity', 'obscene',
               'identity_attack', 'insult', 'threat']
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


def load_data():
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_train.csv'))
    # Preprocess data
    for column in IDENTITY_COLUMNS + ['target']:
        train_df[column] = train_df[column].apply(
            lambda x: 1 if x >= 0.5 else 0)

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

    comments = []
    for t in text_train:
        comments.append(sent_tokenize(t))

    x_train = np.zeros((len(comments), MAX_SENTS, MAX_SENT_LEN),
                       dtype='int32')
    for i, sents in enumerate(comments):
        for j, sent in enumerate(sents):
            if j >= MAX_SENTS:
                continue
            tokens = tk.texts_to_sequences(sent)
            k = 0
            for idx in tokens:
                if len(idx) == 0:
                    continue
                if k < MAX_SENT_LEN and idx[0] < VOCAB_SIZE:
                    x_train[i, j, k] = idx[0]
                    k += 1

    embedding_matrix = generate_weights(
        load_glove_weights(os.path.join(BASE_DIR, 'glove.6B.300d.txt')),
        tk.word_index,
        VOCAB_SIZE - 1,
    )

    # load test data
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'preprocessed_test.csv'))
    text_test = test_df['comment_text'].astype(str).values
    test_comments = []
    for t in text_test:
        test_comments.append(sent_tokenize(t))

    x_test = np.zeros((len(test_comments), MAX_SENTS, MAX_SENT_LEN),
                      dtype='int32')
    for i, sents in enumerate(test_comments):
        for j, sent in enumerate(sents):
            if j >= MAX_SENTS:
                continue
            tokens = tk.texts_to_sequences(sent)
            k = 0
            for idx in tokens:
                if len(idx) == 0:
                    continue
                if k < MAX_SENT_LEN and idx[0] < VOCAB_SIZE:
                    x_test[i, j, k] = idx[0]
                    k += 1

    return x_train, y_train, y_aux_train, test_df.id, x_test, \
        embedding_matrix, sample_weights


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatibl|e with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al.
    [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with
    return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight('{}_W'.format(self.name),
                                 (input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name),
                                     (input_shape[-1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight('{}_u'.format(self.name),
                                 (input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training
        # the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small
        # positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def load_model(weights, hidden_size=100):
    REG_PARAM = 1e-13
    l2_reg = regularizers.l2(REG_PARAM)

    word_input = Input(shape=(MAX_SENT_LEN,), dtype='float32')
    word_embedding = Embedding(weights.shape[0], weights.shape[1],
                               weights=[weights], input_length=MAX_SENT_LEN,
                               trainable=False)(word_input)
    word_lstm = Bidirectional(
        LSTM(hidden_size, return_sequences=True, kernel_regularizer=l2_reg)
    )(word_embedding)
    word_dense = TimeDistributed(
        Dense(256, kernel_regularizer=l2_reg)
    )(word_lstm)
    word_att = AttentionWithContext()(word_dense)
    word_encoder = Model(inputs=word_input, outputs=word_att)

    # Sentence attention model
    sent_input = Input(shape=(MAX_SENTS, MAX_SENT_LEN), dtype='float32')
    sent_encoder = TimeDistributed(word_encoder)(sent_input)
    sent_lstm = Bidirectional(
        LSTM(512, return_sequences=True, kernel_regularizer=l2_reg)
    )(sent_encoder)
    sent_dense = TimeDistributed(
        Dense(256, kernel_regularizer=l2_reg))(sent_lstm)
    sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
    out = Dense(1, activation='sigmoid')(sent_att)
    aux_out = Dense(len(AUX_COLUMNS), activation='sigmoid')(sent_att)
    model = Model(inputs=sent_input, outputs=[out, aux_out])
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
    hidden_size = 512

    # load data
    x_train, y_train, y_aux_train, test_id,\
        x_test, weights, sample_weights = load_data()

    checkpoint = keras.callbacks.ModelCheckpoint(
        'han_bias_model.h5', save_best_only=True)
    es = keras.callbacks.EarlyStopping(patience=3)
    model = load_model(weights, hidden_size)
    history = model.fit(
        x_train, [y_train, y_aux_train],
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[es, checkpoint],
        sample_weight=[sample_weights.values, np.ones_like(sample_weights)]
    )

    # evaluation
    model.load_weights('han_bias_model.h5')
    test_preds = model.predict(x_test)

    submission = pd.DataFrame.from_dict({
        'id': test_id,
        'prediction': test_preds
    })
    submission.to_csv('han_bias_submission.csv', index=False)
