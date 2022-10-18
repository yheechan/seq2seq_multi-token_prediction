import torch
import os

def graphModel(dataloader, model, writer):
    # Again, grab a single mini-batch of images
    dataiter = iter(dataloader)
    prefix, postfix, labels = dataiter.next()

    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(model.cpu(), (prefix, postfix, labels))
    writer.flush()

    print('uploaded model graph to tensorboard!')

def saveModel(fn, project_nm, prefix_pack, postfix_pack, attn_pack):

    os.makedirs('../model/'+fn+'/'+project_nm+'/')

    encoder_mod_idx = 0
    decoder_mod_idx = 1
    encoder_opt_idx = 2
    decoder_opt_idx = 3

    attn_mod_idx = 0
    attn_opt_idx = 1


    prefix_pack[encoder_mod_idx].cpu()
    postfix_pack[encoder_mod_idx].cpu()
    prefix_pack[decoder_mod_idx].cpu()
    postfix_pack[decoder_mod_idx].cpu()

    attn_pack[attn_mod_idx].cpu()

    torch.save(prefix_pack[encoder_mod_idx], '../model/'+fn+'/'+project_nm+'/prefix_encoder.pt')
    torch.save(postfix_pack[encoder_mod_idx], '../model/'+fn+'/'+project_nm+'/postfix_encoder.pt')
    torch.save(prefix_pack[decoder_mod_idx], '../model/'+fn+'/'+project_nm+'/prefix_decoder.pt')
    torch.save(postfix_pack[decoder_mod_idx], '../model/'+fn+'/'+project_nm+'/postfix_decoder.pt')

    torch.save(attn_pack[attn_mod_idx], '../model/'+fn+'/'+project_nm+'/attention.pt')

def getModel(fn):
    model = torch.load('../model/'+fn+'.pt')
    return model