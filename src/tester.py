import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import pandas as pd

def test(
    test_dataloader=None,
    model=None,
    loss_fn=None,
    device=None
):
    

    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.to(device)


    # Put the model to evaluating mode
    model.eval()


    # Tracking variables
    fin_loss = []
    fin_acc = []


    # For each batch in our validation set...
    for step, batch in enumerate(test_dataloader):

        loss = 0
        all_loss = 0


        # Load batch to GPU
        prefix, postfix, labels = tuple(t.to(device) for t in batch)


        # Compute logits
        with torch.no_grad():


            results = model(prefix, postfix, labels, 0.0)

            # add loss for each token predicted
            for i in range(results.shape[0]):
                loss = loss_fn(results[i], labels[:,i])
                all_loss += loss

                # calculate accuracy
                preds = results[i].argmax(1).flatten()
                acc = (preds == labels[:,i]).cpu().numpy().mean() * 100
                fin_acc.append(acc)

                # keep loss
                fin_loss.append(loss.item())


    # Compute the average accuracy and loss over the validation set.
    fin_loss = np.mean(fin_loss)
    fin_acc = np.mean(fin_acc)

    
    print('test loss: ', fin_loss)
    print('test acc: ', fin_acc)

    
    return fin_loss, fin_acc