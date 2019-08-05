import mxnet.ndarray as nd
import mxnet.gluon as gluon
"""
    # This file contains the attention network working with text embedding
"""
class Attention(gluon.Block):
    def __init__(self, seq_length, num_hidden, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.seq_length = seq_length
        with self.name_scope():
            self.fc1 = gluon.nn.Dense(seq_length, activation='relu')

    def forward(self, hidden):
        h_f = hidden[:, :, 0:self.num_hidden]
        h_b = hidden[:, :, self.num_hidden:2 * self.num_hidden]
        H = (h_f + h_b).transpose((0, 2, 1))
        M = nd.tanh(H)
        alpha = nd.softmax(self.fc1(M)).reshape((0, 0, -1))
        r = nd.batch_dot(H, alpha)
        return nd.tanh(r)


class LSTM(gluon.Block):
    def __init__(self, vocab_size, seq_length, num_embed, num_hidden, num_layers, dropout, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.seq_length = seq_length
        with self.name_scope():
            self.encoder = gluon.nn.Embedding(vocab_size, num_embed)
            self.LSTM1 = gluon.rnn.LSTM(num_hidden, num_layers, layout='NTC', bidirectional=True)
            self.dropout = gluon.nn.Dropout(dropout)
            self.attention = Attention(seq_length, num_hidden)
            self.fc1 = gluon.nn.Dense(2)

    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self.LSTM1(emb, hidden)
        output = self.dropout(output)
        output = self.attention(output)
        output = self.fc1(output)
        return nd.softmax(output, axis=1), hidden

    def begin_state(self, *args, **kwargs):
        return self.LSTM1.begin_state(*args, **kwargs)