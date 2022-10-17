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

        # embedded_output = [batch_size, embed_dim] --> [batch_size, data_length(token numbers), embed_dim]
        embedded_output = self.embedding(input)
        print('--after embeding--')
        print(embedded_output.shape)
        
        # output = [batch_size, data_length(token numbers), embed_dim] --> [batch_size, data_length(token numbers), hidden_size*2]
        # hidden = [batch_size, data_length(token numbers), embed_dim] --> [2, batch_size, hidden_size]
        # cell = [batch_size, data_length(token numbers), embed_dim] --> [2, batch_size, hidden_size] 
        output, (hidden, cell) = self.lstm(embedded_output)
        print('--after lstm output, hidden, cell--')
        print(output.shape)
        print(hidden.shape)
        print(cell.shape)

        return output, (hidden, cell)
    
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

        # self.attn = nn.Linear((self.hidden_size * 2 * 2) + embed_dim, self.max_length)

        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.lstm = nn.LSTM(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            dropout = 0.0,
            batch_first = True,
            bidirectional = True
        )

        # self.dropout = nn.Dropout(self.dropout)

        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, encoder_outputs):

        print('--enter decoder input, encoder_output, hidden, cell--')
        print(input.shape)
        print(encoder_outputs.shape)
        print(hidden.shape)
        print(cell.shape)

        # embedded_output = [batch_size, 1] --> [batch_size, 1, embed_dim]
        embedded_output = self.embedding(input)
        print('--decoder embeding shape--')
        print(embedded_output.shape)

        # output = [batch_size, 1, embed_dim] --> [batch_size, 1, hidden_size*2] 
        # hidden = [batch_size, 1, embed_dim] --> [2, batch_size, hidden_size]
        # cell = [batch_size, 1, embed_dim] --> [2, batch_size, hidden_size]
        # hidden and cell contains the final state for each data in the batch
        output, (hidden, cell) = self.lstm(embedded_output, (hidden, cell))
        print('--after lstm output, hidden, cell--')
        print(output.shape)
        print(hidden.shape)
        print(cell.shape)

        # change hidden state of decoder embedding of single token 
        # to match hidden states in each state of encoder
        # encoder_output --> [batch_size, data_length(token numbers), embed_dim]
        # output from lstm decoder --> [batch_size, embed_dim, 1]
        output_perm = output.permute(0, 2, 1)
        print('--after permute output--')
        print(encoder_outputs.shape)
        print(output_perm.shape)

        # attn_score = [batch_size, data_length(token numbers), 1]
        attn_score = torch.bmm(encoder_outputs, output_perm)
        print('--bmm shape--')
        print(attn_score.shape)

        # [batch_size, data_length(token numbers), 1]
        attn_dist = F.softmax(attn_score)
        print('--attn_dist--')
        print(attn_dist.shape)

        # [batch_size, data_length(token numbers), hidden_size*2]
        weighted = torch.bmm(attn_dist, output)
        print('--weighted--')
        print(weighted.shape)


        # [batch_size, data_length(token numbers)]
        attn_value = torch.sum(weighted, 2)
        print('--attn value--')
        print(attn_value.shape)

        return attn_value, output


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
        print('--prefix shape--')
        print(prefix.shape)
        print('--label shape--')
        print(labels.shape)

        batch_size = labels.shape[0]
        label_len = labels.shape[1]

        # encoder [batch_size, embed_dim] -->
        # output = [batch_size, data_length(token numbers), hidden_size*2]
        # hidden = [2, batch_size, hidden_size]
        prefix_output, (prefix_hidden, prefix_cell) = self.prefixEncoder(prefix)
        previous_prefix_hidden = prefix_output


        # encoder [batch_size, embed_dim] -->
        # output = [batch_size, data_length(token numbers), hidden_size*2]
        # hidden = [2, batch_size, hidden_size]
        postfix_output, (postfix_hidden, postfix_cell) = self.postfixEncoder(postfix)
        previous_postfix_hidden = postfix_output

        # gives the first token for each labels in batch
        # input = [batch_size, 1] (containing the 0st token)
        input = labels[:,0].unsqueeze(1)
        print('--first input to decoder shape--')
        print(input.shape)

        for i in range(1, label_len):

            # output --> [batch_size, data_length(token numbers)]
            # hidden --> [batch_size, 1, hidden_size*2]
            attn_prefix_output, previous_prefix_hidden = self.decoder(input, previous_prefix_hidden)

            # output --> [batch_size, data_length(token numbers)]
            # hidden --> [batch_size, 1, hidden_size*2]
            attn_postfix_output, previous_postfix_hidden = self.decoder(input, previous_postfix_hidden)

            # depending on teacher forcing
            # input = labels[:,i]

            break