import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
from keras.preprocessing import text
from keras.engine.topology import Layer
import keras.layers as L
from keras.models import Model
from keras import initializers as initializers
from keras import backend as K

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize

from glovevectorizer import load_glove_weights, generate_weights

# BASE_DIR = '/home/kwu14/data/cs584_course_project'
BASE_DIR = '../data/'

VOCAB_SIZE = 10000

MAX_SENTS = 43
MAX_SENT_LEN = 300

AUX_COLUMNS = ['severe_toxicity', 'obscene',
               'identity_attack', 'insult', 'threat']
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


class AttentionLayer(Layer):
    """
    Hierarchial Attention Layer as described by Hierarchical
    Attention Networks for Document Classification(2016)
    - Yang et. al.
    Source: 
    https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    Theano backend
    """
    def __init__(self, attention_dim=100, return_coefficients=False, **kwargs):
        # Initializer
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform') # initializes values with uniform distribution
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Builds all weights
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]

        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, hit, mask=None):
        # Here, the actual calculation is done
        uit = K.bias_add(K.dot(hit, self.W),self.b)
        uit = K.tanh(uit)
        
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        
        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = hit * ait
        
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]


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


def load_model(weights, hidden_size=100):
    # Words level attention model
    word_input = L.Input(shape=(MAX_SENT_LEN,), dtype='int32')
    word_sequences = L.Embedding(weights.shape[0], weights.shape[1], weights=[weights], input_length=MAX_SENT_LEN, trainable=False, name='word_embedding')(word_input)
    word_gru = L.Bidirectional(L.GRU(hidden_size, return_sequences=True))(word_sequences)
    word_dense = L.Dense(100, activation='relu', name='word_dense')(word_gru)
    word_att, word_coeffs = AttentionLayer(100, True, name='word_attention')(word_dense)
    wordEncoder = Model(inputs=word_input, outputs=word_att)

    # Sentence level attention model
    sent_input = L.Input(shape=(MAX_SENTS, MAX_SENT_LEN), dtype='int32', name='sent_input')
    sent_encoder = L.TimeDistributed(wordEncoder, name='sent_linking')(sent_input)
    sent_gru = L.Bidirectional(L.GRU(50, return_sequences=True))(sent_encoder)
    sent_dense = L.Dense(100, activation='relu', name='sent_dense')(sent_gru)
    sent_att, sent_coeffs = AttentionLayer(100, return_coefficients=True, name='sent_attention')(sent_dense)
    sent_drop = L.Dropout(0.5, name='sent_dropout')(sent_att)
    preds = L.Dense(1, activation='sigmoid', name='output')(sent_drop)

    # Model compile
    model = Model(sent_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(wordEncoder.summary())
    print(model.summary())
    return model


if __name__ == "__main__":
    # hyper-paramters
    batch_size = 1024
    epochs = 50
    hidden_size = 128

    # load data
    x_train, y_train, y_aux_train, test_id,\
        x_test, weights, sample_weights = load_data()

    checkpoint = keras.callbacks.ModelCheckpoint(
        'han_bias_model.h5', save_best_only=True, verbose=1)
    es = keras.callbacks.EarlyStopping(patience=3, verbose=1)
    model = load_model(weights, hidden_size)
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[es, checkpoint],
        verbose=2
        # sample_weight=[sample_weights.values, np.ones_like(sample_weights)]
    )

    # evaluation
    model.load_weights('han_bias_model.h5')
    test_preds = model.predict(x_test)

    submission = pd.DataFrame.from_dict({
        'id': test_id,
        'prediction': test_preds
    })
    submission.to_csv('han_bias_submission.csv', index=False)
