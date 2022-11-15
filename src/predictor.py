import torch
import torch.nn.functional as F
import torch.utils.data as tud
from tqdm.auto import tqdm
import copy


def predictNoBeam(
    prefix,
    postfix,
    model=None,
    device=None
):


    # ********************* INSTANTIATE MODEL INPUT DATA *********************

    # [batch_size (1), token_length (64)]
    prefix = torch.tensor(prefix).unsqueeze(dim=0)
    postfix = torch.tensor(postfix).unsqueeze(dim=0)

    labels = torch.zeros(1, 10).to(device).long()

    model = model.to(device)
    prefix = prefix.to(device)
    postfix = postfix.to(device)
    labels = labels.to(device)




    # ********************* PREDICT TOKEN SEQUENCE *********************

    # [token_labels (10 tokens), batch_size (1), token choices (214 tokens choices)]
    results = model(prefix, postfix, labels)


    tok_list = []
    for i in range(results.shape[0]):

        # the token with highest probability
        preds = results[i].argmax(1).flatten()

        # append each token to list
        tok_list.append(preds.item())
    
    
    return tok_list







def myBeamStart(
    model,
    prefix,
    postfix,
    device=None,
    predictions=10,
    beam_width=2
):
    # ********************* INSTANTIATE MODEL INPUT DATA *********************

    # [batch_size (1), token_length (64)]
    prefix = torch.tensor(prefix).unsqueeze(dim=0)
    postfix = torch.tensor(postfix).unsqueeze(dim=0)

    # [batch_size, token_label]
    # Y = torch.full((PREFIX.shape[0], 1), 213).to(device).long()


    total_token_seq = list()
    total_seq_score = list()
    token_stack = list()
    score_stack = list()

    labels = torch.tensor(copy.deepcopy([0])).unsqueeze(dim=0)


    model = model.to(device)
    prefix = prefix.to(device)
    postfix = postfix.to(device)
    labels = labels.to(device)


    beamSearch(
        model,

        prefix,
        postfix,
        labels,
 
        total_token_seq,
        total_seq_score,
        token_stack,
        score_stack,
        0,
        beam_width=beam_width
    )

    torch_scores = torch.FloatTensor(total_seq_score)
    sorted, indices = torch.sort(torch_scores, descending=True)

    pred_results = []
    for i in range(5):
        ordered_score_idx = indices[i].item()
        pred_results.append(total_token_seq[ordered_score_idx])
    
    return pred_results




def beamSearch(
    model,
    prefix,
    postfix,
    labels,
    total_token_seq,
    total_seq_score,
    token_stack,
    score_stack,
    limit,
    beam_width=2
):

    sorted, indices = beamed(model, prefix, postfix, labels)


    for i in range(beam_width):

        end_value = indices[i].item()
        end_score = sorted[i].item()

        # if limit == 10 or end_value == 0:
        if limit == 10:
            total_score_tensor = torch.sum(torch.tensor(copy.deepcopy(score_stack)))
            total_seq_score.append(total_score_tensor)

            total_token_seq.append(copy.deepcopy(token_stack))
            break
        else:
            limit += 1

            token_stack.append(end_value)
            token_stack.append(0)
            score_stack.append(end_score)
 
            labels = torch.tensor(copy.deepcopy(token_stack)).unsqueeze(dim=0).to(
                next(model.parameters()).device
            ).long()
            token_stack.pop()

            beamSearch(
                model,
                prefix,
                postfix,
                labels,
                total_token_seq,
                total_seq_score,
                token_stack,
                score_stack,
                limit
            )

            token_stack.pop()
            score_stack.pop()
            limit -= 1



def beamed(model, prefix, postfix, labels):
    with torch.no_grad():
        logits = model.forward(prefix, postfix, labels, beam=True)
        probs = F.softmax(logits[-1], dim=1).squeeze(dim=0)
        sorted, indices = torch.sort(probs, descending=True)
    return sorted, indices







def beam_search(
    model, 
    PREFIX, 
    POSTFIX,
    device = None,
    predictions = 10,
    beam_width = 5,
    batch_size = 50, 
    progress_bar = 0
):
    """
    Implements Beam Search to compute the output with the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width. 

    progress_bar: int 
        Shows a tqdm progress bar, useful for tracking progress with large tensors. Ranges from 0 to 2.

    Returns
    -------
    Y: LongTensor of shape (examples, length + predictions)
        The output sequences.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """

    # ********************* INSTANTIATE MODEL INPUT DATA *********************

    # [batch_size (1), token_length (64)]
    PREFIX = torch.tensor(PREFIX).unsqueeze(dim=0)
    POSTFIX = torch.tensor(POSTFIX).unsqueeze(dim=0)

    Y = torch.full((PREFIX.shape[0], 1), 213).to(device).long()

    model = model.to(device)
    PREFIX = PREFIX.to(device)
    POSTFIX = POSTFIX.to(device)
    Y = Y.to(device)

    with torch.no_grad():


        # Y = torch.ones(PREFIX.shape[0], 1).to(next(model.parameters()).device).long()
        

        # The next command can be a memory bottleneck, can be controlled with the batch 
        # size of the predict method.

        # [label_len (10 labels), batch_size, output_size (214 token choices)]
        next_probabilities = model.forward(PREFIX, POSTFIX, Y, beam=True)
        
        next_probabilities = next_probabilities[-1, :, :]

        vocabulary_size = next_probabilities.shape[-1]

        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1)\
        .topk(k = beam_width, axis = -1)

        Y = Y.repeat((beam_width, 1))
        next_chars = next_chars.reshape(-1, 1)
        Y = torch.cat((Y, next_chars), axis = -1)
        # Y = next_chars

        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)

        for i in predictions_iterator:
            dataset = tud.TensorDataset(
                PREFIX.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1),
                POSTFIX.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1),
                Y)

            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(loader)

            if progress_bar > 1:
                iterator = tqdm(iterator)

            for prefix, postfix, y in iterator:
                # print(y)
                probs = model.forward(prefix, postfix, y, beam=True)[-1, :, :].log_softmax(-1)
                next_probabilities.append(probs)


            next_probabilities = torch.cat(next_probabilities, axis = 0)
            # print('\n--cat probs--')
            # print(next_probabilities.shape)

            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            # print('\n--reshape probs--')
            # print(next_probabilities.shape)

            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            # print('\n-- + probs--')
            # print(probabilities.shape)

            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(k = beam_width, axis = -1)

            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            # print('\n--next_chars--')
            # print(next_chars)
            # print()
            # print()

            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(Y.shape[0] // beam_width, device = PREFIX.device).unsqueeze(-1) * beam_width
            Y = Y[best_candidates].flatten(end_dim = -2)
            Y = torch.cat((Y, next_chars), axis = 1)

        return Y.reshape(-1, beam_width, Y.shape[-1]), probabilities