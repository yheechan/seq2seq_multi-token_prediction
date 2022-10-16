import seq2seq_model as sm
import torch.optim as optim
import torch.nn as nn

def initialize_model(
    vocab_size=None,
    embed_dim=200,
    hidden_size=100,
    num_classes=2,
    rnn_layers=3,
    num_filters=[100, 100, 100],
    kernel_sizes=[8, 8, 8],
    dropout=0.2,
    learning_rate=0.001,
    weight_decay=1e-4,
    model_name="RNN",
    optim_name="Adam",
    loss_fn_name="CEL",
    pretrained_model=None,
    freeze_embedding=False,
    device=None,):

    if model_name == "RNN":

        model = sm.C_rnn(
            input_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            n_classes=num_classes,
            n_layers=rnn_layers,
            dropout=dropout,
            pretrained_embedding=pretrained_model,
            freeze_embedding=freeze_embedding
        )

    model.to(device)

    if optim_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay)
    
    if loss_fn_name == "CEL":
        loss_fn = nn.CrossEntropyLoss()
    
    return model, optimizer, loss_fn