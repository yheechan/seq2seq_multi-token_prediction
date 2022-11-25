import torch
import os

def graphModel(dataloader, model, writer, device):
    # Again, grab a single mini-batch of images
    dataiter = iter(dataloader)
    prefix, postfix, labels = dataiter.next()

    model = model.to(device)
    prefix = prefix.to(device)
    postfix = postfix.to(device)
    labels = labels.to(device)

    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(model, (prefix, postfix, labels))
    writer.flush()

    print('uploaded model graph to tensorboard!')

def saveModel(fn, project_nm, model):

    # os.makedirs('../model/'+fn+'/'+project_nm+'/')

    model.cpu()
    torch.save(model, '../model/'+fn+'/'+project_nm+'.pt')


def getModel(fn, project_nm):
    model = torch.load('../model/'+fn+'/'+project_nm+'.pt')

    return model