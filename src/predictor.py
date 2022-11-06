from tkinter import W
import torch
import torch.nn.functional as F
import math
from collections import defaultdict
import copy

def predict(prefix, postfix, prefix_pack, postfix_pack, attn_pack):

    prefix = torch.tensor(prefix).unsqueeze(dim=0)
    postfix = torch.tensor(postfix).unsqueeze(dim=0)

    # --------------------------------------------------------

    encoder_mod_idx = 0
    decoder_mod_idx = 1
    encoder_opt_idx = 2
    decoder_opt_idx = 3

    attn_mod_idx = 0
    attn_opt_idx = 1


    # --------------------------------------------------------

    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    # prefix_pack[encoder_mod_idx].to(device)
    # postfix_pack[encoder_mod_idx].to(device)
    # prefix_pack[decoder_mod_idx].to(device)
    # postfix_pack[decoder_mod_idx].to(device)

    # attn_pack[attn_mod_idx].to(device)


    prefix_pack[encoder_mod_idx].cpu()
    postfix_pack[encoder_mod_idx].cpu()
    prefix_pack[decoder_mod_idx].cpu()
    postfix_pack[decoder_mod_idx].cpu()

    attn_pack[attn_mod_idx].cpu()


    prefix_pack[encoder_mod_idx].eval()
    postfix_pack[encoder_mod_idx].eval()
    prefix_pack[decoder_mod_idx].eval()
    postfix_pack[decoder_mod_idx].eval()

    attn_pack[attn_mod_idx].eval()

    # --------------------------------------------------------

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
    input = torch.full((1, 1), 213).cpu()

    # --------------------------------------------------------

    total_token_seq = list()
    total_seq_score = list()
    token_stack = list()
    score_stack = list()

    bean_search(
        input,
        0,

        prefix_pack[decoder_mod_idx],
        postfix_pack[decoder_mod_idx],

        prefix_state,
        postfix_state,

        attn_pack[attn_mod_idx],

        encoder_prefix_hiddens,
        encoder_postfix_hiddens,

        total_token_seq,
        total_seq_score,
        token_stack,
        score_stack
    )

    torch_scores = torch.FloatTensor(total_seq_score)
    sorted, indices = torch.sort(torch_scores, descending=True)

    pred_results = []
    for i in range(5):
        pred_results.append(total_token_seq[indices[i]])

    return pred_results




def bean_search(
    input,
    limit,

    prefix_decoder,
    postfix_decoder,

    prefix_state,
    postfix_state,

    attn_model,

    encoder_prefix_hiddens,
    encoder_postfix_hiddens,

    total_token_seq,
    total_seq_score,
    token_stack,
    score_stack
):

    limit += 1


    value = input[0].item()

    if limit == 10 or value == 1:

        score = log_score(score_stack)

        total_token_seq.append( copy.deepcopy(token_stack) )
        total_seq_score.append( score )

    
    else:
        prefix_state, postfix_state, \
        sorted, indices =   return_predictions(
                                input,

                                prefix_decoder,
                                postfix_decoder,

                                prefix_state,
                                postfix_state,

                                attn_model,

                                encoder_prefix_hiddens,
                                encoder_postfix_hiddens,
                            )
        
        
        for i in range(2):
            token_stack.append(indices[i].item())
            score_stack.append(sorted[i].item())

            input = torch.full((1, 1), indices[i].item()).cpu()

            bean_search(
                input,
                limit,

                prefix_decoder,
                postfix_decoder,

                prefix_state,
                postfix_state,

                attn_model,

                encoder_prefix_hiddens,
                encoder_postfix_hiddens,

                total_token_seq,
                total_seq_score,
                token_stack,
                score_stack
            )

            token_stack.pop()
            score_stack.pop()


def log_score(list):
    mul_score = 1

    for i in list:
        mul_score *= i
    
    log_score = math.log(mul_score)


    return log_score

def return_predictions(
    input,

    prefix_decoder,
    postfix_decoder,

    prefix_state,
    postfix_state,

    attn_model,

    encoder_prefix_hiddens,
    encoder_postfix_hiddens,
):
    # [batch_size, single token, hidden_size*2]
    decoder_prefix_hiddens, prefix_state = prefix_decoder(input, prefix_state)
    decoder_postfix_hiddens, postfix_state = postfix_decoder(input, postfix_state)

    # [batch_size, output_size]
    logits = attn_model(
        encoder_prefix_hiddens,
        encoder_postfix_hiddens,
        decoder_prefix_hiddens,
        decoder_postfix_hiddens
    )

    answer = logits.argmax(1).unsqueeze(1)

    probs = F.softmax(logits, dim=1).squeeze(dim=0)

    sorted, indices = torch.sort(probs, descending=True)

    # print('---------')
    # print(answer)

    # for i in range(5):
    #     print(str(indices[i].item()) + ': ' + str(sorted[i].item()) + '%')

    # if answer == 1:
    #     print('done')
    #     break

    # input = answer

    return prefix_state, postfix_state, sorted, indices