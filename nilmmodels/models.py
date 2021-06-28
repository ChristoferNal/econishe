import torch
from torch import nn
from torchnlp.nn import Attention


class LinearDropoutLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        super(LinearDropoutLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.linear(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ConvLayer, self).__init__()

        left, right = kernel_size // 2, kernel_size // 2
        if kernel_size % 2 == 0:
            right -= 1
        padding = (left, right, 0, 0)

        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class WGRU(nn.Module):

    def __init__(self, dropout=0, lr=None):
        super(WGRU, self).__init__()
        self.architecture_name = "WGRU"

        self.drop = dropout
        self.lr = lr

        self.conv1 = ConvLayer(1, 16, kernel_size=4, dropout=self.drop)

        self.b1 = nn.GRU(16, 64, batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)
        self.b2 = nn.GRU(128, 256, batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)

        self.dense1 = LinearDropoutLayer(512, 128, self.drop)
        self.dense2 = LinearDropoutLayer(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv1(x)
        # x (aka output of conv1) shape is [batch_size, out_channels=16, window_size-kernel+1]
        # x must be in shape [batch_size, seq_len, input_size=output_size of prev layer]
        # so we have to change the order of the dimensions
        x = x.permute(0, 2, 1)
        x = self.b1(x)[0]
        x = self.b2(x)[0]
        # we took only the first part of the tuple: output, h = gru(x)

        # Next we have to take only the last hidden state of the last b2gru
        # equivalent of return_sequences=False
        x = x[:, -1, :]
        # x = x.reshape(-1, 512 * 50)

        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out


class SAED(nn.Module):

    def __init__(self, window_size, mode='dot', hidden_dim=16,
                 num_heads=1, dropout=0, lr=None):
        '''
         mode(str): 'dot' or 'general'--> additive
             default is 'dot' (additive attention not supported yet)
         ***in order for the mhattention to work, embed_dim should be dividable
         to num_heads (embed_dim is the hidden dimension inside mhattention
         '''
        super(SAED, self).__init__()
        self.architecture_name = "SAED"
        if num_heads > hidden_dim:
            num_heads = 1
            print('WARNING num_heads > embed_dim so it is set equal to 1')
        else:
            while hidden_dim % num_heads:
                if num_heads > 1:
                    num_heads -= 1
                else:
                    num_heads += 1

        self.drop = dropout
        self.lr = lr
        self.mode = 'dot'

        self.conv = ConvLayer(1, hidden_dim,
                              kernel_size=4,
                              dropout=self.drop)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
        #                                             num_heads=num_heads,
        #                                             dropout=self.drop)
        self.attention = Attention(window_size, attention_type=mode)
        self.bgru = nn.GRU(hidden_dim, 64,
                           batch_first=True,
                           bidirectional=True,
                           dropout=self.drop)
        self.dense = LinearDropoutLayer(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)

        x, _ = self.attention(x, x)
        x = x.permute(0, 2, 1)

        x = self.bgru(x)[0]
        # we took only the first part of the tuple: output, h = gru(x)

        # Next we have to take only the last hidden state of the last b1gru
        # equivalent of return_sequences=False
        x = x[:, -1, :]
        # x = x.reshape(-1, 128 * 50)

        x = self.dense(x)
        out = self.output(x)
        return out


class Seq2Point(nn.Module):

    def __init__(self, window_size, dropout=0, lr=None):
        super(Seq2Point, self).__init__()
        self.architecture_name = "Seq2Point"
        self.drop = dropout
        self.lr = lr
        self.last_conv_output = 10

        self.dense_input = self.last_conv_output * window_size  # 50 is the out_features of last CNN1

        self.conv = nn.Sequential(
            ConvLayer(1, 30, kernel_size=10, dropout=self.drop),
            ConvLayer(30, 40, kernel_size=8, dropout=self.drop),
            ConvLayer(40, 50, kernel_size=6, dropout=self.drop),
            ConvLayer(50, 50, kernel_size=5, dropout=self.drop),
            ConvLayer(50, self.last_conv_output, kernel_size=5, dropout=self.drop),
            nn.Flatten()
        )
        self.dense = LinearDropoutLayer(self.dense_input, 1024, self.drop)
        self.output = nn.Linear(1024, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dense(x)
        out = self.output(x)
        return out