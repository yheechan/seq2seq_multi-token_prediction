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
        
        # output = [batch_size, data_length(token numbers), embed_dim] --> [batch_size, data_length(token numbers), hidden_size*2]
        # hidden = [batch_size, data_length(token numbers), embed_dim] --> [2, batch_size, hidden_size]
        # cell = [batch_size, data_length(token numbers), embed_dim] --> [2, batch_size, hidden_size] 
        output, (hidden, cell) = self.lstm(embedded_output)

        return output, (hidden, cell)
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()

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

    def forward(self, input, end_state):

        # embedded_output = [batch_size, 1] --> [batch_size, 1, embed_dim]
        embedded_output = self.embedding(input)

        # output = [batch_size, 1, embed_dim] --> [batch_size, 1, hidden_size*2] 
        # hidden = [batch_size, 1, embed_dim] --> [2, batch_size, hidden_size]
        # cell = [batch_size, 1, embed_dim] --> [2, batch_size, hidden_size]
        # hidden and cell contains the final state for each data in the batch
        output, (hidden, cell) = self.lstm(embedded_output, end_state)

        return output, (hidden, cell)


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        dropout
    ):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.fc = nn.Linear(
            self.hidden_size*2*2*2,
            self.output_size
        )

        self.dp = nn.Dropout(self.dropout)

    def forward(
        self,
        encoder_prefix_hiddens,
        encoder_postfix_hiddens,
        decoder_prefix_hiddens,
        decoder_postfix_hiddens
    ):


        # [batch_size, hidden_size*2, single token]
        decoder_prefix_hiddens_perm = decoder_prefix_hiddens.permute(0, 2, 1)
        decoder_postfix_hiddens_perm = decoder_postfix_hiddens.permute(0, 2, 1) 


        # [batch_size, token numbers token, 1]
        prefix_attn_score = torch.bmm(encoder_prefix_hiddens, decoder_prefix_hiddens_perm)
        postfix_attn_score = torch.bmm(encoder_postfix_hiddens, decoder_postfix_hiddens_perm)


        # [batch_size, token numbers, 1]
        prefix_attn_dist = F.softmax(prefix_attn_score, dim=1)
        postfix_attn_dist = F.softmax(postfix_attn_score, dim=1)


        # [batch_size, token numbers, hidden_size*2]
        prefix_weighted = torch.bmm(prefix_attn_dist, decoder_prefix_hiddens)
        postfix_weighted = torch.bmm(postfix_attn_dist, decoder_postfix_hiddens)


        prefix_weighted_perm = prefix_weighted.permute(0, 2, 1)
        postfix_weighted_perm = postfix_weighted.permute(0, 2, 1)

        # [batch_size, hidden_size*2]
        prefix_attn_value = torch.sum(prefix_weighted_perm, 2)
        postfix_attn_value = torch.sum(postfix_weighted_perm, 2)


        decoder_prefix_hiddens_squeeze = decoder_prefix_hiddens.squeeze(1)
        decoder_postfix_hiddens_squeeze = decoder_postfix_hiddens.squeeze(1)


        # [batch_size, hidden_size*2*2]
        prefix_cat = torch.cat((decoder_prefix_hiddens_squeeze, prefix_attn_value), 1)
        postfix_cat = torch.cat((decoder_postfix_hiddens_squeeze, postfix_attn_value), 1)


        # [batch_size, hidden_size*2*2*2]
        final_cat = torch.cat((prefix_cat, postfix_cat), 1)


        final_tanh = torch.tanh(final_cat)


        # [batch_size, output_size]
        y = self.fc(self.dp(F.relu(final_tanh)))


        # [batch_size, output_size]
        result = F.log_softmax(y, dim=1)

        return result





class MySeq2Seq(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        hidden_size=200,
        n_layers=2,
        output_size=215,
        dropout=0.3,
        max_length=66,
        input_size=215,
        device=None,
        loss_fn=None,
        teacher_forcing=True
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
        self.loss_fn=loss_fn
        self.teacher_forcing=teacher_forcing

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

        self.decoder = Decoder(
           embed_dim = self.embed_dim,
           hidden_size = self.hidden_size,
           n_layers = self.n_layers,
           output_size = self.output_size,
           dropout = self.dropout,
           max_length = self.max_length,
           input_size = self.input_size,
           device = self.device 
        )

        self.attn = Attention(
            hidden_size = self.hidden_size,
            output_size = self.output_size,
            dropout = self.dropout
        )


    def forward(
        self,
        prefix,
        postfix,
        labels,
    ):

        batch_size = labels.shape[0]
        label_len = labels.shape[1]

        # encoder [batch_size, embed_dim] -->
        # output = [batch_size, token numbers, hidden_size*2]
        # hidden = [2, batch_size, hidden_size]
        encoder_prefix_hiddens, prefix_state = self.prefixEncoder(prefix)


        # encoder [batch_size, embed_dim] -->
        # output = [batch_size, token numbers, hidden_size*2]
        # hidden = [2, batch_size, hidden_size]
        encoder_postfix_hiddens, postfix_state = self.postfixEncoder(postfix)


        # gives the first token for each labels in batch
        # input = [batch_size, 1] (containing the 0st token)
        # input = labels[:,0].unsqueeze(1)
        input = torch.full((batch_size, 1), 213).to(self.device)


        # [label_len, batch_size, output_size]
        outputs = torch.zeros(
            label_len, batch_size, self.output_size
        ).to(self.device)

        loss = 0
        loss_box = []
        acc_box = []

        for i in range(-1, label_len-1, 1):

            # [batch_size, single token, hidden_size*2*2]
            decoder_prefix_hiddens, prefix_state = self.decoder(input, prefix_state)
            decoder_postfix_hiddens, postfix_state = self.decoder(input, postfix_state)

            # [batch_size, output_size]
            result = self.attn(
                encoder_prefix_hiddens,
                encoder_postfix_hiddens,
                decoder_prefix_hiddens,
                decoder_postfix_hiddens
            )

            # [batch_size]
            expected = labels[:, i+1]
            # [batch_size, output_size]
            outputs[i+1] = result 

            loss += self.loss_fn(result, expected)
            loss_box.append(loss.item()/1+(i+1))
            
            preds = result.argmax(1).flatten()
            acc = (preds == expected).cpu().numpy().mean() * 100
            acc_box.append(acc)

            # depending on teacher forcing
            # [bach_size, 1]
            if self.teacher_forcing:
                input = labels[:,i+1].unsqueeze(1)
            else:
                input = result.argmax(1).unsqueeze(1)

        return outputs.permute(1, 0, 2), loss, loss_box, acc_box 