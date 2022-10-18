import seq2seq_model as sm
import torch.optim as optim
import torch.nn as nn

def initialize_model(
    learning_rate=0.001,
    weight_decay=0.0,
    embed_dim=128,
    hidden_size=200,
    n_layers=2,
    output_size=215,
    dropout=0.3,
    max_length=66,
    input_size=215,
    device=None,
    model_name='seq2seq',
    optim_name='Adam',
    loss_fn_name='CEL',
    teacher_forcing=True
):


    if loss_fn_name == 'CEL':
        loss_fn = nn.CrossEntropyLoss()


    if model_name == 'seq2seq':

        model = sm.MySeq2Seq(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            output_size=output_size,
            dropout=dropout,
            max_length=max_length,
            input_size=input_size,
            device=device,
            loss_fn=loss_fn,
            teacher_forcing=teacher_forcing
        )

    model.to(device)
    

    if optim_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


    return model, optimizer, loss_fn