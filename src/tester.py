import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import pandas as pd

def test(
    test_dataloader=None,
    device=None, 
    loss_fn=None,
    prefix_pack=None,
    postfix_pack=None,
    attn_pack=None
):
    

    teacher_forcing = False


    encoder_mod_idx = 0
    decoder_mod_idx = 1
    encoder_opt_idx = 2
    decoder_opt_idx = 3

    attn_mod_idx = 0
    attn_opt_idx = 1


    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    prefix_pack[encoder_mod_idx].to(device)
    postfix_pack[encoder_mod_idx].to(device)
    prefix_pack[decoder_mod_idx].to(device)
    postfix_pack[decoder_mod_idx].to(device)

    attn_pack[attn_mod_idx].to(device)


    prefix_pack[encoder_mod_idx].eval()
    postfix_pack[encoder_mod_idx].eval()
    prefix_pack[decoder_mod_idx].eval()
    postfix_pack[decoder_mod_idx].eval()

    attn_pack[attn_mod_idx].eval()


    # Tracking variables
    fin_loss = []
    fin_acc = []


    # For each batch in our validation set...
    for step, batch in enumerate(test_dataloader):


        # Load batch to GPU
        prefix, postfix, label = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():


            batch_size = label.shape[0]
            label_len = label.shape[1]


            # Perform a forward pass. This will return logits.
    
            # encoder [batch_size, embed_dim] -->
            # output = [batch_size, token numbers, hidden_size*2]
            # hidden = [2, batch_size, hidden_size]
            encoder_prefix_hiddens, prefix_state = prefix_pack[encoder_mod_idx](prefix)
    
    
            # encoder [batch_size, embed_dim] -->
            # output = [batch_size, token numbers, hidden_size*2]
            # hidden = [2, batch_size, hidden_size]
            encoder_postfix_hiddens, postfix_state = postfix_pack[encoder_mod_idx](postfix)


            # gives the first token for each labels in batch
            # input = [batch_size, 1] (containing the 0st token)
            # input = labels[:,0].unsqueeze(1)
            input = torch.full((batch_size, 1), 213).to(device)


            # [label_len, batch_size, output_size]
            outputs = torch.zeros(
                label_len, batch_size, 216
            ).to(device)


            loss = 0


            for i in range(-1, label_len-1, 1):

                # [batch_size, single token, hidden_size*2]
                decoder_prefix_hiddens, prefix_state = prefix_pack[decoder_mod_idx](input, prefix_state)
                decoder_postfix_hiddens, postfix_state = postfix_pack[decoder_mod_idx](input, postfix_state)

                # [batch_size, output_size]
                result = attn_pack[attn_mod_idx](
                    encoder_prefix_hiddens,
                    encoder_postfix_hiddens,
                    decoder_prefix_hiddens,
                    decoder_postfix_hiddens
                )

                # [batch_size]
                expected = label[:, i+1]
                # [batch_size, output_size]
                outputs[i+1] = result 

                loss += loss_fn(result, expected)
                fin_loss.append(loss.item()/1+(i+1))
                
                preds = result.argmax(1).flatten()
                acc = (preds == expected).cpu().numpy().mean() * 100
                fin_acc.append(acc)

                # depending on teacher forcing
                # [bach_size, 1]
                if teacher_forcing:
                    input = label[:,i+1].unsqueeze(1)
                else:
                    input = result.argmax(1).unsqueeze(1) 


    # Compute the average accuracy and loss over the validation set.
    fin_loss = np.mean(fin_loss)
    fin_acc = np.mean(fin_acc)

    
    print('test loss: ', fin_loss)
    print('test acc: ', fin_acc)

    
    return fin_loss, fin_acc