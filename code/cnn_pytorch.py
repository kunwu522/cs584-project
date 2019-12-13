import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 output_size, vocab_size, weights):
        super(TextCNN, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, input_size)
        self.word_embedding.weight = nn.Parameter(weights, requires_grad=False)

        if hidden_size % 2 != 0:
            raise ValueError(
                'Expect even number of hidden_size, got ', hidden_size)

        self.conv_layers = []
        for i in range(0, hidden_size):
            kernel_size = int(i / 2) + 2
            conv = nn.Conv2d(1, 1, (kernel_size, input_size),
                             stride=1, bias=True)
            if torch.cuda.is_available():
                conv = conv.cuda()

            self.conv_layers.append(conv)

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.word_embedding(x)
        outputs = [torch.tanh(conv(x)) for conv in self.conv_layers]
        outputs = [torch.topk(out.squeeze(), 1, dim=1)[0] for out in outputs]
        outputs = torch.cat(outputs, dim=1)
        return torch.sigmoid(self.linear(outputs))


class TextCNN2(nn.Module):

    def __init__(self, embedding_size, output_size, batch_size,
                 vocab_size, weights):
        super(TextCNN2, self).__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_embedding.weight = nn.Parameter(weights, requires_grad=False)

        self.kernel_size_4 = 4
        self.kernel_size_3 = 3
        self.kernel_size_2 = 2
        self.conv1 = nn.Conv2d(
            1, 1, (self.kernel_size_4, embedding_size), stride=1, bias=True)
        self.conv2 = nn.Conv2d(
            1, 1, (self.kernel_size_4, embedding_size), stride=1, bias=True)
        self.conv3 = nn.Conv2d(
            1, 1, (self.kernel_size_3, embedding_size), stride=1, bias=True)
        self.conv4 = nn.Conv2d(
            1, 1, (self.kernel_size_3, embedding_size), stride=1, bias=True)
        self.conv5 = nn.Conv2d(
            1, 1, (self.kernel_size_2, embedding_size), stride=1, bias=True)
        self.conv6 = nn.Conv2d(
            1, 1, (self.kernel_size_2, embedding_size), stride=1, bias=True)

        self.fc = nn.Linear(6, 1, bias=True)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.word_embedding(x)

        out1 = torch.tanh(self.conv1(x))
        out2 = torch.tanh(self.conv1(x))
        out3 = torch.tanh(self.conv3(x))
        out4 = torch.tanh(self.conv4(x))
        out5 = torch.tanh(self.conv5(x))
        out6 = torch.tanh(self.conv6(x))

        out1 = torch.topk(out1.view(batch_size, -1), 1, dim=1)[0]
        out2 = torch.topk(out2.view(batch_size, -1), 1, dim=1)[0]
        out3 = torch.topk(out3.view(batch_size, -1), 1, dim=1)[0]
        out4 = torch.topk(out4.view(batch_size, -1), 1, dim=1)[0]
        out5 = torch.topk(out5.view(batch_size, -1), 1, dim=1)[0]
        out6 = torch.topk(out6.view(batch_size, -1), 1, dim=1)[0]

        x = torch.cat((out1, out2, out3, out4, out5, out6), dim=1)

        x = self.fc(x)
        return torch.sigmoid(x)


class TextCNN3(nn.Module):
    ''' CNN for text classification model, which have 3 convolutional layers,
        the kernel size is 2, 3 and 4 respectively.

        Input:
          vocab_size: scaler, the size of vocabulary
          embedding_size: scaler, the size of word embedding
          hidden_size: number of output channel of each conv layer, normally
                       set 2 or 3
          output_size: the number of classes
    '''

    def __init__(self, embedding_size, hidden_size,
                 output_size, vocab_size, weights):
        super(TextCNN3, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(weights, requires_grad=False)

        self.conv1 = nn.Conv1d(embedding_size, hidden_size, kernel_size=2)
        self.conv2 = nn.Conv1d(embedding_size, hidden_size, kernel_size=3)
        self.conv3 = nn.Conv1d(embedding_size, hidden_size, kernel_size=4)

        self.linear = nn.Linear(3 * hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        if output_size == 1:
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        h1 = torch.max(self.conv1(x), dim=2)[0]
        h2 = torch.max(self.conv2(x), dim=2)[0]
        h3 = torch.max(self.conv3(x), dim=2)[0]

        h = torch.cat((h1, h2, h3), dim=1)
        h = self.dropout(h)
        return self.output_layer(self.linear(h))
