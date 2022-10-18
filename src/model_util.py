import torch

def graphModel(dataloader, model, writer):
    # Again, grab a single mini-batch of images
    dataiter = iter(dataloader)
    prefix, postfix, labels = dataiter.next()

    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(model.cpu(), (prefix, postfix, labels))
    writer.flush()

    print('uploaded model graph to tensorboard!')

def saveModel(fn, model):
    model.cpu()
    torch.save(model, '../model/'+fn+'.pt')

def getModel(fn):
    model = torch.load('../model/'+fn+'.pt')
    return model