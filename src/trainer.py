import random
import numpy as np
import time
import torch

def set_seed(seed_value=43):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(
    epochs=40,
    title='testing',
    writer=None,
    train_dataloader=None,
    val_dataloader=None,
    device=None,
    model=None,
    optimizer=None,
    loss_fn=None
):

    best_accuracy = 0
    model.to(device)
    
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^8} | {'Val Acc':^6} | {'Elapsed':^6}")
    print("-"*80)

    for epoch_i in range(epochs):

        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()

        # Put the model into the training mode
        model.train()
        tot_train_acc = []
        tot_train_loss = []

        torch.autograd.set_detect_anomaly(True)

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            prefix, postfix, label = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            # model.zero_grad()
            optimizer.zero_grad()

            # Perform a forward pass. This will return logits.
            logits, loss, tot_train_loss, tot_train_acc = model(
                prefix, postfix, label
            )

            # print('--b4 logits and label--')
            # print(logits.shape)
            # print(label.shape)

            # output_dim = logits.shape[-1]

            # print('--output_dim--')
            # print(output_dim)

            # logits = logits.view(-1, output_dim)
            # label = label.view(-1)

            # # Compute loss and accumulate the loss values
            # print('--after logits and label--')
            # print(logits.shape)
            # print(label.shape)

            # loss = loss_fn(logits, label)
            # tot_train_loss.append(loss.item())

            # # Get the predictions
            # preds = torch.argmax(logits, dim=1).flatten()

            # # Calculate the accuracy rate
            # accuracy = (preds == label).cpu().numpy().mean() * 100
            # tot_train_acc.append(accuracy)

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()
        


        # Calculate the average loss over the entire training data
        avg_train_loss = np.mean(tot_train_loss)
        avg_train_acc = np.mean(tot_train_acc)

        # =======================================
        #               Evaluation
        # =======================================

        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_acc = evaluate(device=device,
                                         model=model,
                                         loss_fn=loss_fn,
                                         val_dataloader=val_dataloader)

            # Track the best accuracy
            if val_acc > best_accuracy:
                best_accuracy = val_acc

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {avg_train_acc:^10.6f} | {val_loss:^8.6f} | {val_acc:^6.2f} | {time_elapsed:^6.2f}")

        writer.add_scalars(title + '-Loss',
                { 'Train' : avg_train_loss, 'Validation' : val_loss },
                epoch_i + 1)

        writer.add_scalars(title + '-Accuracy',
                    { 'Train' : avg_train_acc, 'Validation' : val_acc },
                    epoch_i + 1)
    
    writer.flush()

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

def evaluate(
    device=None, 
    model=None,
    loss_fn=None,
    val_dataloader=None):

    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """


    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.to(device)
    model.eval()

    # Tracking variables
    val_loss = []
    val_accuracy = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        prefix, postfix, label = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            # Perform a forward pass. This will return logits.
            logits, loss, val_loss, val_accuracy = model(
                prefix, postfix, label
            )

        # # Compute loss
        # loss = loss_fn(logits, label)
        # val_loss.append(loss.item())

        # # Get the predictions
        # preds = torch.argmax(logits, dim=1).flatten()

        # # Calculate the accuracy rate
        # accuracy = (preds == label).cpu().numpy().mean() * 100
        # val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy