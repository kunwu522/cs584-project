import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Embedding, LSTM


class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, layers_num, bidirectional,
                 vocab_size, dropout, embedding_length, weights):
        super(LSTMClassifier, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.layers_num = 2 * layers_num if bidirectional else layers_num

        self.word_embedding = Embedding(vocab_size, embedding_length)
        self.word_embedding.weights = nn.Parameter(
            weights, requires_grad=False)
        self.lstm = LSTM(embedding_length, hidden_size,
                         num_layers=layers_num, dropout=dropout,
                         bidirectional=bidirectional)
        # self.dropout = Dropout(dropout)
        self.label = nn.Linear(self.layers_num * hidden_size, output_size)

    def forward(self, x):
        x = self.word_embedding(x)
        x = x.permute(1, 0, 2)
        h_0 = Variable(torch.rand(self.layers_num,
                                  x.size(1), self.hidden_size))
        c_0 = Variable(torch.rand(self.layers_num,
                                  x.size(1), self.hidden_size))

        if torch.cuda.is_available():
            h_0 = h_0.cuda(torch.device('cuda'))
            c_0 = c_0.cuda(torch.device('cuda'))

        output, (final_hidden_state, final_cell_state) = self.lstm(
            x, (h_0, c_0))
        out = final_hidden_state.permute(1, 0, 2)
        out = out.contiguous().view(out.size(0), out.size(1) * out.size(2))
        out = self.label(out)
        return torch.sigmoid(out)
