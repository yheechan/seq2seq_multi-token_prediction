import random
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

        self.hidden_fc = nn.Linear(
            self.hidden_size,
            self.hidden_size//2
        )

        self.cell_fc = nn.Linear(
            self.hidden_size,
            self.hidden_size//2
        )

        self.dp = nn.Dropout(self.dropout)

    def forward(self, input):

        # embedded_output = [batch_size, embed_dim] --> [batch_size, data_length(token numbers), embed_dim]
        embedded_output = self.embedding(input)
        
        # output = [batch_size, data_length(token numbers), embed_dim] --> [batch_size, data_length(token numbers), hidden_size*2]
        # hidden = [batch_size, data_length(token numbers), embed_dim] --> [2, batch_size, hidden_size]
        # cell = [batch_size, data_length(token numbers), embed_dim] --> [2, batch_size, hidden_size] 
        output, (hidden, cell) = self.lstm(embedded_output)


        # hidden = [2, batch_size, hidden_size/2]
        # cell = [2, batch_size, hidden_size/2] 
        hidden = self.hidden_fc(self.dp(F.relu(hidden)))
        cell = self.cell_fc(self.dp(F.relu(cell)))

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

        self.lstm = nn.LSTM(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            dropout = 0.0,
            batch_first = True,
            bidirectional = True
        )

    def forward(self, input, end_state):


        # embedded_output = [batch_size, 1] --> [batch_size, 1, embed_dim]
        embedded_output = self.embedding(input)
        # print('--embedd_output--')
        # print(embedded_output.shape)

        # output = [batch_size, 1, embed_dim] --> [batch_size, 1, hidden_size*2] 
        # hidden = [batch_size, 1, embed_dim] --> [2, batch_size, hidden_size]
        # cell = [batch_size, 1, embed_dim] --> [2, batch_size, hidden_size]
        # hidden and cell contains the final state for each data in the batch
        output, (hidden, cell) = self.lstm(embedded_output, end_state)

        # print('--final--')
        # print(output.shape)
        # print(hidden.shape)
        # print(cell.shape)

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
            self.hidden_size*2*2,
            self.output_size
        )

        self.dp = nn.Dropout(self.dropout)

    def forward(
        self,
        encoder_hiddens,
        decoder_end_state
    ):

        # encoder_hiddens = [batch_size, token_numbers (128), hidden_size*2]
        # decoder_end_state = [batch_size, hidden*2, 1]


        # [batch_size, token_numbers (128), 1]
        attn_score = torch.bmm(encoder_hiddens, decoder_end_state)


        # [batch_size, token_numbers (128), 1]
        attn_dist = F.softmax(attn_score, dim=1)


        # [batch_size, 1, token_numbers (128)]
        attn_dist_perm = attn_dist.permute(0, 2, 1)


        # [batch_size, 1, hidden_size*2]
        attn_weighted = torch.bmm(attn_dist_perm, encoder_hiddens)


        # [batch_size, hidden_size*2]
        attn_value = torch.sum(attn_weighted, 1)


        # [batch_size, hidden_size*2]
        decoder_end_state_squeeze = decoder_end_state.squeeze(2)


        # [batch_size, hidden_size*2*2]
        final_cat = torch.cat((decoder_end_state_squeeze, attn_value), 1)


        final_tanh = torch.tanh(final_cat)


        # [batch_size, output_size]
        y = self.fc(self.dp(F.relu(final_tanh)))


        # [batch_size, output_size]
        result = F.log_softmax(y, dim=1)

        return result





class MySeq2Seq(nn.Module):
    def __init__(
        self,
        embed_dim=100,
        hidden_size=200,
        n_layers=1,
        output_size=214,
        dropout=0.3,
        max_length=64,
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


    def forward(self, prefix, postfix, labels, teacher_forcing_ratio):

        batch_size = labels.shape[0]
        label_len = labels.shape[1]




        # ********************* INPUT PREFIX & POSTFIX TO ENCODER *********************

        # encoder [batch_size, embed_dim] -->
        # output = [batch_size, token_numbers (64), hidden_size*2]
        # end_state (for each) = [2, batch_size, hidden_size]
        encoder_prefix_hiddens, prefix_end_state = self.prefixEncoder(prefix)


        # encoder [batch_size, embed_dim] -->
        # output = [batch_size, token_numbers (64), hidden_si
        # end_state (for each) = [2, batch_size, hidden_size]
        encoder_postfix_hiddens, postfix_end_state = self.postfixEncoder(postfix)


        # [batch_size, token_numbers (128), hidden_size*2]
        encoder_hiddens = torch.cat((encoder_prefix_hiddens, encoder_postfix_hiddens), 1)





        # ********************* CONCATENATE END STATE FROM BOTH ENCODERS *********************

        # hidden = [2, batch_size, hidden_size/2]
        # cell = [2, batch_size, hidden_size/2]
        (prefix_hidden, prefix_cell) = prefix_end_state
        (postfix_hidden, postfix_cell) = postfix_end_state


        # concatenate end state from prefix and postfix encoder
        hidden = torch.cat((prefix_hidden, postfix_hidden), 2)
        cell = torch.cat((prefix_cell, postfix_cell), 2)


        # hidden = [2, batch_size, hidden_size]
        # cell = [2, batch_size, hidden_size]
        end_state = (hidden, cell)




        # ********************* SET [INPUT | OUTPUT] VARIABLE *********************
        # ** teacher_forcing for each batch **

        # gives the first token for each labels in batch
        # 213 is BOS
        # [batch_size, 1]
        input = torch.full((batch_size, 1), 213).to(self.device)


        # [label_len (10 labels), batch_size, output_size (214 token choices)]
        outputs = torch.zeros(
            label_len, batch_size, self.output_size
        ).to(self.device)


        # loss = 0
        # loss_box = []
        # acc_box = []

        # python random.random() return floating number between 0 and 1
        teacher_forcing = True if random.random() < teacher_forcing_ratio else False




        # ********************* PREDICT EACH TOKEN BY SEQUENCE *********************
        # ** first input given by 213 as BOS **

        for i in range(-1, label_len-1, 1):




            # ********************* INPUT ENCODER END STATE TO DECODER *********************

            # [batch_size, single token, hidden_size*2]
            # [2, batch_size, hidden_size] for each hidden and cell in end_state
            decoder_hiddens, end_state = self.decoder(input, end_state)
            # decoder_postfix_hiddens, postfix_state = postfix_pack[decoder_mod_idx](input, prefix_state, postfix_state)

            (hidden, cell) = end_state

            # [batch_size, hidden_size*2]
            decoder_end_hidden = torch.cat((hidden[0], hidden[1]), 1)

            # [batch_size, hidden_size*2, 1]
            decoder_end_hidden = decoder_end_hidden.unsqueeze(2)




            # ********************* GENERATE RESULT FROM ENCODER HIDDENS & DECODER END STATE *********************

            # [batch_size, output_size] (214 choices of token)
            result = self.attn(encoder_hiddens, decoder_end_hidden)


            # save result to outputs for each token labels (10 labels)
            # [batch_size, output_size]
            outputs[i+1] = result 


            if teacher_forcing:
                input = labels[:,i+1].unsqueeze(1)
            else:
                input = result.argmax(1).unsqueeze(1)

        return outputs 