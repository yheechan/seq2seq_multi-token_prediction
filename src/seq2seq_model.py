from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(
        self,
        embed_dim,
        hidden_size,
        n_layers,
        dropout,
        input_size,
        device
    ):
        super(Encoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device

        self.embedding = nn.Embedding(self.input_size, self.embed_dim)

        self.lstm = nn.LSTM(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            dropout = self.dropout,
            batch_first = True,
            bidirectional = True
        )

    def forward(self, input):

        # embedded_output = [1000, 64] --> [1000, 64, embed_dim]
        embedded_output = self.embedding(input)
        print('==after embeding')
        print(embedded_output.shape)
        
        # output = [1000, 64, embed_dim] --> [1000, 64, hidden_size]
        # hidden = [1000, 64, embed_dim] --> [2, 1000, 30]
        # cell = [1000, 64, embed_dim] --> [2, 1000, 30]
        output, (hidden, cell) = self.lstm(embedded_output)
        print('==after lstm')
        print(output.shape)
        print(hidden.shape)
        print(cell.shape)

        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_size,
        n_layers,
        output_size,
        dropout,
        max_length,
        input_size,
        device
    ):
        super(AttnDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length
        self.input_size = input_size
        self.device = device

        self.embedding = nn.Embedding(self.input_size, self.embed_dim)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.lstm = nn.LSTM(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            dropout = 0.0,
            batch_first = True,
            bidirectional = True
        )

        self.dropout = nn.Dropout(self.dropout)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedding_output = self.embedding(input).view(1, 1, -1)
        embedding_output = self.dropout(embedding_output)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedding_output[0], hidden[0]), 1)),
            dim=1
        )

        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedding_output[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class MySeq2Seq(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        hidden_size=30,
        n_layers=1,
        output_size=214,
        dropout=0.0,
        max_length=65,
        input_size=214,
        device=None
    ):

        super(MySeq2Seq, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length
        self.input_size = input_size
        self.device = device

        self.prefixEncoder = Encoder(
            embed_dim = self.embed_dim,
            hidden_size = self.hidden_size,
            n_layers = self.n_layers,
            dropout = self.dropout,
            input_size = self.input_size,
            device = self.device
        )

        self.postfixEncoder = Encoder(
            embed_dim = self.embed_dim,
            hidden_size = self.hidden_size,
            n_layers = self.n_layers,
            dropout = self.dropout,
            input_size = self.input_size,
            device = self.device
        )

        self.decoder = AttnDecoder(
           embed_dim = self.embed_dim,
           hidden_size = self.hidden_size,
           n_layers = self.n_layers,
           output_size = self.output_size,
           dropout = self.dropout,
           max_length = self.max_length,
           input_size = self.input_size,
           device = self.device 
        )

    def forward(self, prefix, postfix, labels):
        print(prefix.shape)
        print(postfix.shape)
        print(labels.shape)

        # encoder [1000, 64] -->
        # output = [1000, 64, hidden_size]
        # hidden = [2, 1000, 30]
        output, hidden = self.prefixEncoder(prefix)