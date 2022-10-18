import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import pandas as pd

def test(
    test_dataloader=None,
    device=None, 
    model=None,
    title='tests'):
    
    loss_fn = nn.CrossEntropyLoss()
    
    model.to(device)
    model.eval()

    tot_loss = []
    tot_pred = torch.empty(0)
    tot_label = torch.empty(0)

    tot_pred = tot_pred.to(device)
    tot_label = tot_label.to(device)

    for batch in test_dataloader:
        prefix, postfix, labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(prefix, postfix, labels, False)
        
        loss = loss_fn(logits, labels)
        tot_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        tot_pred = torch.cat((tot_pred, preds))
        tot_label = torch.cat((tot_label, labels))
    
    fin_loss = np.mean(tot_loss)
    fin_acc = (tot_pred == tot_label).cpu().numpy().mean() * 100
    
    print('test loss: ', fin_loss)
    print('test acc: ', fin_acc)

    results = metrics.classification_report(tot_label.cpu(), tot_pred.cpu(), output_dict=True)
    results_df = pd.DataFrame.from_dict(results).transpose()
    results_df.to_excel('../result/'+title+'.xlsx', sheet_name='sheet1')

    print('saved precision and recall results to file!')
    
    return fin_loss, fin_acc