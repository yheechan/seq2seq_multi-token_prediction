import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

def data_loader(
    train_prefix,
    train_postfix,
    val_prefix,
    val_postfix,
    test_prefix,
    test_postfix,
    train_label,
    val_label,
    test_label,
    batch_size=50):

    
    train_prefix, train_postfix, \
    val_prefix, val_postfix, \
    test_prefix, test_postfix, \
    train_label, val_label, test_label =\
        tuple(torch.tensor(data) for data in 
            [
                train_prefix, train_postfix,
                val_prefix, val_postfix,
                test_prefix, test_postfix,
                train_label, val_label, test_label
            ])
    
    # Create DataLoader for training data
    train_data = TensorDataset(train_prefix, train_postfix, train_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_prefix, val_postfix, val_label)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, drop_last=True)

    # Create DataLoader for training data
    test_data = TensorDataset(test_prefix, test_postfix, test_label)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader 