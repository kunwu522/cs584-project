import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from lstm import LSTMClassifier
from cnn_pytorch import TextCNN3
from glovevectorizer import load_glove_weights, generate_weights

BASE_DIR = '/home/kwu14/data/cs584_course_project'
# BASE_DIR = '../data/'

DATA_SIZE = 200000

VOCAB_SIZE = 20000
MAX_LEN = 250
EMBEDDING_SIZE = 300

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


class CommentsDataset(Dataset):
    def __init__(self, data, weight=None):
        super(CommentsDataset, self).__init__()
        self.texts, self.labels = data
        self.weight = weight

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.weight is None:
            return self.texts[idx], self.labels[idx]
        else:
            return self.texts[idx], self.labels[idx], self.weight[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int,
                        default=0, help='The id of GPU.')
    parser.add_argument('-m', '--model', type=str, help='The model')
    parser.add_argument('-b', '--bias', action='store_true', help='')
    args = parser.parse_args()
    print('Params: ', args)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    epochs = 50
    learning_rate = 0.001
    batch_size = 500
    hidden_size = 100

    bias = args.bias

    x_train, y_train, test_id, x_test, \
        embedding_weights, sample_weights = load_data()

    # sample_weights = torch.Tensor(sample_weights)
    embedding_weights = torch.Tensor(embedding_weights)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2)
    train_dataset = CommentsDataset((x_train, y_train), sample_weights)
    val_dataset = CommentsDataset((x_val, y_val))

    if args.model == 'lstm':
        model = LSTMClassifier(1, hidden_size, 2, True, VOCAB_SIZE, 0.5,
                               EMBEDDING_SIZE, embedding_weights)
    elif args.model == 'cnn':
        model = TextCNN3(EMBEDDING_SIZE, 3, 1, VOCAB_SIZE, embedding_weights)

    if torch.cuda.is_available():
        model.cuda(torch.device('cuda'))

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for t in range(epochs):

        total_epoch_acc = 0
        total_epoch_loss = 0
        model.train()
        for x_train, y_train, weight in tqdm(train_loader, desc=f'Epoch {t}'):
            x_train = x_train.long()
            y_train = y_train.float().unsqueeze(1)
            weight = weight.unsqueeze(1)

            if torch.cuda.is_available():
                x_train = x_train.cuda(torch.device('cuda'))
                y_train = y_train.cuda(torch.device('cuda'))
                weight = weight.cuda(torch.device('cuda'))

            optimizer.zero_grad()
            predict_y = model(x_train)
            if bias:
                loss = F.binary_cross_entropy(predict_y, y_train, weight)
            else:
                loss = F.binary_cross_entropy(predict_y, y_train)
            loss.backward()
            optimizer.step()

            num_corrects = torch.eq(
                torch.where(predict_y.cpu() >= 0.5, torch.ones(predict_y.size()),
                            torch.zeros(predict_y.size())),
                y_train.cpu()).sum()
            acc = 100.0 * num_corrects / batch_size

            total_epoch_acc += acc.item()
            total_epoch_loss += loss.cpu().item()

        total_val_acc = 0.
        total_val_loss = 0.
        for x_val, y_val in val_loader:
            x_val = x_val.long()
            y_val = y_val.float().unsqueeze(1)

            if torch.cuda.is_available():
                x_val = x_val.cuda(torch.device('cuda'))
                y_val = y_val.cuda(torch.device('cuda'))

            model.eval()
            with torch.no_grad():
                y_pred_val = model(x_val)
                val_loss = F.binary_cross_entropy(y_pred_val, y_val)

            num_corrects = torch.eq(
                torch.where(predict_y.cpu() >= 0.5, torch.ones(predict_y.size()),
                            torch.zeros(predict_y.size())).cpu(),
                y_train.cpu()).sum()
            acc = 100.0 * num_corrects / batch_size
            total_val_acc += acc.item()
            total_val_loss += val_loss.cpu().item()

        train_loss = total_epoch_acc / len(train_loader)
        train_acc = total_epoch_acc / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        val_acc = total_val_acc / len(val_loader)

        print('#### Epoch {},'.format(t,),
              'Training loss: {:.4f}, acc: {:.4f}'.format(
                  train_loss, train_acc),
              'Val loss: {:.4f}, acc:{:.4f}'.format(val_loss, val_acc))

    x_test = torch.LongTensor(x_test)
    if torch.cuda.is_available():
        x_test = x_test.cuda(torch.device('cuda'))

    test_set = CommentsDataset((x_test, None))
    test_loader = DataLoader(test_set, batch_size=1000)
    predict_Y = []

    with torch.no_grad():
        for x_test, _ in test_loader:
            if torch.cuda.is_available():
                x_test = x_test.cuda(torch.device('cuda'))

            y_pred_test = model(x_test)

            predict_Y.extend(torch.where(predict_y.cpu(), torch.ones(
                predict_y.size(), torch.zeros(predict_y.size()))).tolist())

    submission = pd.DataFrame.from_dict({
        'id': test_id,
        'prediction': np.array(predict_Y)
    })
    bias = 'bias' if args.bias else ''
    submission.to_csv(
        f'../outputs/{args.model}_{bias}_submission.csv',
        index=False
    )
    print('Finished, save submission in ')
