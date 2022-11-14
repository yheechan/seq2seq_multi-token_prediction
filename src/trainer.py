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
    teacher_forcing_ratio=0.5,
    train_dataloader=None,
    val_dataloader=None,
    model=None,
    loss_fn=None,
    optimizer=None,
    device=None,
):


    best_accuracy = 0

    # put model to device
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


        # Put the model into training mode
        model.train()


        tot_train_acc = []
        tot_train_loss = []


        for step, batch in enumerate(train_dataloader):


            # Load batch to GPU
            prefix, postfix, labels = tuple(t.to(device) for t in batch)


            # Zero out any previously calculated gradients
            # model.zero_grad()
            optimizer.zero_grad()

            # [label_len (10 labels), batch_size, output_size (214 token choices)]
            results = model(prefix, postfix, labels, teacher_forcing_ratio)

            loss = 0
            all_loss = 0

            # add loss for each token predicted
            for i in range(results.shape[0]):
                loss = loss_fn(results[i], labels[:,i])
                all_loss += loss

                # calculate accuracy
                preds = results[i].argmax(1).flatten()
                acc = (preds == labels[:,i]).cpu().numpy().mean() * 100
                tot_train_acc.append(acc)

                # calculate loss
                tot_train_loss.append(loss.item())


            torch.autograd.set_detect_anomaly(True)
            all_loss.backward(retain_graph=True)
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
            val_loss, val_acc = evaluate(
                val_dataloader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                device=device
            )


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
    val_dataloader=None,
    model=None,
    loss_fn=None,
    device=None
):

    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """


    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.to(device)


    # Put the model to evaluating mode
    model.eval()


    # Tracking variables
    val_loss = []
    val_accuracy = []



    # For each batch in our validation set...
    for step, batch in enumerate(val_dataloader):

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
                val_accuracy.append(acc)

                # keep loss
                val_loss.append(loss.item())


    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy