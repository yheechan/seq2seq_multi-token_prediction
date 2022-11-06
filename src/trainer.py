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
    loss_fn=None,
    prefix_pack=None,
    postfix_pack=None,
    attn_pack=None,
    teacher_forcing_ratio=100.0
):

    teacher_forcing = True

    best_accuracy = 0

    encoder_mod_idx = 0
    decoder_mod_idx = 1
    encoder_opt_idx = 2
    decoder_opt_idx = 3

    attn_mod_idx = 0
    attn_opt_idx = 1

    prefix_pack[encoder_mod_idx].to(device)
    postfix_pack[encoder_mod_idx].to(device)
    prefix_pack[decoder_mod_idx].to(device)
    postfix_pack[decoder_mod_idx].to(device)

    attn_pack[attn_mod_idx].to(device)
    
    
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
        prefix_pack[encoder_mod_idx].train()
        postfix_pack[encoder_mod_idx].train()
        prefix_pack[decoder_mod_idx].train()
        postfix_pack[decoder_mod_idx].train()

        attn_pack[attn_mod_idx].train()


        tot_train_acc = []
        tot_train_loss = []


        for step, batch in enumerate(train_dataloader):


            # Load batch to GPU
            prefix, postfix, label = tuple(t.to(device) for t in batch)


            # Zero out any previously calculated gradients
            # model.zero_grad()
            prefix_pack[encoder_opt_idx].zero_grad()
            postfix_pack[encoder_opt_idx].zero_grad()
            prefix_pack[decoder_opt_idx].zero_grad()
            postfix_pack[decoder_opt_idx].zero_grad()

            attn_pack[attn_opt_idx].zero_grad()


            batch_size = label.shape[0]
            label_len = label.shape[1]


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

            # 213 is BOS
            input = torch.full((batch_size, 1), 213).to(device)


            # [label_len, batch_size, output_size]
            outputs = torch.zeros(
                label_len, batch_size, 214
            ).to(device)


            loss = 0


            teacher_forcing = True if random.random() < teacher_forcing_ratio else False


            for i in range(-1, label_len-1, 1):

                # [batch_size, single token, hidden_size*2]
                decoder_prefix_hiddens, prefix_state = prefix_pack[decoder_mod_idx](input, prefix_state)
                decoder_postfix_hiddens, postfix_state = postfix_pack[decoder_mod_idx](input, postfix_state)

                # [batch_size, output_size]
                result = attn_pack[attn_mod_idx](
                    encoder_prefix_hiddens,
                    encoder_postfix_hiddens,
                    decoder_prefix_hiddens,
                    decoder_postfix_hiddens
                )

                # [batch_size]
                expected = label[:, i+1]
                # [batch_size, output_size]
                outputs[i+1] = result 

                loss += loss_fn(result, expected)
                tot_train_loss.append(loss.item()/1+(i+1))
                
                preds = result.argmax(1).flatten()
                acc = (preds == expected).cpu().numpy().mean() * 100
                tot_train_acc.append(acc)

                # depending on teacher forcing
                # [bach_size, 1]
                if teacher_forcing:
                    input = label[:,i+1].unsqueeze(1)
                else:
                    input = result.argmax(1).unsqueeze(1) 


            loss.backward()

            # Update parameters
            prefix_pack[encoder_opt_idx].step()
            postfix_pack[encoder_opt_idx].step()
            prefix_pack[decoder_opt_idx].step()
            postfix_pack[decoder_opt_idx].step()

            attn_pack[attn_opt_idx].step()
        


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
                device=device,
                loss_fn=loss_fn,
                prefix_pack=prefix_pack,
                postfix_pack=postfix_pack,
                attn_pack=attn_pack
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
    device=None,
    loss_fn=None,
    prefix_pack=None,
    postfix_pack=None,
    attn_pack=None
):

    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """


    teacher_forcing = False

    encoder_mod_idx = 0
    decoder_mod_idx = 1
    encoder_opt_idx = 2
    decoder_opt_idx = 3

    attn_mod_idx = 0
    attn_opt_idx = 1


    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    prefix_pack[encoder_mod_idx].to(device)
    postfix_pack[encoder_mod_idx].to(device)
    prefix_pack[decoder_mod_idx].to(device)
    postfix_pack[decoder_mod_idx].to(device)

    attn_pack[attn_mod_idx].to(device)


    prefix_pack[encoder_mod_idx].eval()
    postfix_pack[encoder_mod_idx].eval()
    prefix_pack[decoder_mod_idx].eval()
    postfix_pack[decoder_mod_idx].eval()

    attn_pack[attn_mod_idx].eval()


    # Tracking variables
    val_loss = []
    val_accuracy = []


    # For each batch in our validation set...
    for step, batch in enumerate(val_dataloader):


        # Load batch to GPU
        prefix, postfix, label = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():


            batch_size = label.shape[0]
            label_len = label.shape[1]


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
            input = torch.full((batch_size, 1), 213).to(device)


            # [label_len, batch_size, output_size]
            outputs = torch.zeros(
                label_len, batch_size, 214
            ).to(device)


            loss = 0


            for i in range(-1, label_len-1, 1):

                # [batch_size, single token, hidden_size*2]
                decoder_prefix_hiddens, prefix_state = prefix_pack[decoder_mod_idx](input, prefix_state)
                decoder_postfix_hiddens, postfix_state = postfix_pack[decoder_mod_idx](input, postfix_state)

                # [batch_size, output_size]
                result = attn_pack[attn_mod_idx](
                    encoder_prefix_hiddens,
                    encoder_postfix_hiddens,
                    decoder_prefix_hiddens,
                    decoder_postfix_hiddens
                )

                # [batch_size]
                expected = label[:, i+1]
                # [batch_size, output_size]
                outputs[i+1] = result 

                loss += loss_fn(result, expected)
                val_loss.append(loss.item()/1+(i+1))
                
                preds = result.argmax(1).flatten()
                acc = (preds == expected).cpu().numpy().mean() * 100
                val_accuracy.append(acc)

                # depending on teacher forcing
                # [bach_size, 1]
                if teacher_forcing:
                    input = label[:,i+1].unsqueeze(1)
                else:
                    input = result.argmax(1).unsqueeze(1) 


    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy