import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.init as init


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class cnnClassifier(nn.Module):
    def __init__(self):
        super(cnnClassifier, self).__init__()

        self.classifier = nn.Sequential(

            nn.Linear(512*7*7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )


        self.out = nn.Sequential(

            nn.Linear(512, 11),
            nn.Softmax(dim=1)

        )


    def forward(self, x):

        x = self.classifier(x)
        x = self.out(x)

        return x


class rnnNet(nn.Module):
    def __init__(self, n_layers=2, h_size=512):
        super(rnnNet, self).__init__()

        self.n_layers = n_layers
        self.h_size = h_size

        self.lstm = nn.LSTM(512*7*7, h_size, dropout=0.5, num_layers=n_layers)

        self.classifier = nn.Sequential(

            nn.BatchNorm1d(h_size),
            nn.Linear(h_size, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 11),
            nn.Softmax(dim=1)

        )

    def forward(self, y, seq):
        state = self._init_state(b_size=len(seq))
        pack = torch.nn.utils.rnn.pack_padded_sequence(y, seq, batch_first=False)
        _, (hn,_) = self.lstm(pack, state)

        hidden = hn[-1]
        t = self.classifier(hidden)

        return t, hidden

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01)),
            Variable(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01))
        )

class seq_to_seq(nn.Module):
    def __init__(self, n_layers=2, h_size=512):
        super(seq_to_seq, self).__init__()

        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(512*7*7, h_size, dropout=0.5, num_layers=n_layers)

        self.classifier = nn.Sequential(

            nn.BatchNorm1d(h_size),
            nn.Linear(h_size, 64),
            nn.BatchNorm1d(64),
            #nn.ReLU(inplace=True),
            nn.Linear(64, 11),
            nn.Softmax(dim=1)

        )

    def forward(self, y):
        state = self._init_state(b_size=1)
        #pack = torch.nn.utils.rnn.pack_padded_sequence(y, seq, batch_first=False, enforce_sorted=False)

        z, _ = self.lstm(y, state)

        t = self.classifier(z.squeeze(1))

        return t

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01)),
            Variable(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01))
        )

def weight_init(m):
    '''
    Source: https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5?fbclid=IwAR2JWPwfEoZax04iKmWRF9qD5Y1ZcHkFoUUcL2WQusYM4ke87B7pGE47yx4
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)