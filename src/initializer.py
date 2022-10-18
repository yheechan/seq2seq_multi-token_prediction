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
    device=None
):


    loss_fn = nn.CrossEntropyLoss()


    prefix_encoder = sm.Encoder(
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        input_size=input_size,
        device=device
    )
    prefix_encoder.to(device)


    postfix_encoder = sm.Encoder(
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        input_size=input_size,
        device=device
    )
    postfix_encoder.to(device)


    prefix_decoder = sm.Decoder(
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        output_size=output_size,
        dropout=dropout,
        max_length=max_length,
        input_size=input_size,
        device=device
    )
    prefix_decoder.to(device)


    postfix_decoder = sm.Decoder(
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        output_size=output_size,
        dropout=dropout,
        max_length=max_length,
        input_size=input_size,
        device=device
    )
    postfix_decoder.to(device)


    attn = sm.Attention(
        hidden_size=hidden_size,
        output_size=output_size,
        dropout=dropout
    )
    attn.to(device)
    


    prefix_encoder_optimizer = optim.Adam(
        prefix_encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    postfix_encoder_optimizer = optim.Adam(
        postfix_encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    prefix_decoder_optimzer = optim.Adam(
        prefix_decoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    postfix_decoder_optimzer = optim.Adam(
        postfix_decoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    attn_optimizer = optim.Adam(
        attn.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    prefix_pack = [prefix_encoder, prefix_decoder, prefix_encoder_optimizer, prefix_decoder_optimzer]
    postfix_pack = [postfix_encoder, postfix_decoder, postfix_encoder_optimizer, postfix_decoder_optimzer]
    attn_pack = [attn, attn_optimizer]

    return loss_fn, prefix_pack, postfix_pack, attn_pack