import torch 


def masked_loss(output, target, loss_fxn, mask):
    """
    Computes the masked loss. Only non-masked values contribute to the loss.
    
    Parameters:
    - output: Model's predictions (B, T, output_size)
    - target: Ground truth values (B, T, output_size)
    - mask: Mask tensor where 1 indicates valid data and 0 indicates padding (B, T)
    
    Returns:
    - Loss value averaged over the non-masked elements
    """
    # Compute the loss for all values
    loss = loss_fxn(output, target)  # Shape (B, T, output_size)

    # Apply mask: multiply loss by mask (ensuring we only count valid values)
    loss = loss * mask.unsqueeze(-1)  # Unsqueeze to align with output shape

    # Average over non-masked values
    masked_loss = loss.sum() / mask.sum()

    return masked_loss


def train_RNN(model, dataloader, validationloader, optimizer, loss_fxn, config, device, generator, savepath):
    min_loss = float('inf')
    val_losses = []
    losses = []
    early_stop_loss = config['training']['early_stopping_loss']
    epochs = config['training']['epochs']
    save_name = config['training']['save_path']
    mask_loss = config['training'].get('mask_loss', False)
    clip_val = config['training'].get('grad_clip', None)  # Optional gradient clipping

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for data in dataloader:
            if mask_loss:
                inputs, targets, mask = data[0].to(device), data[1].to(device), data[2].to(device)
            else:
                inputs, targets = data[0].to(device), data[1].to(device)
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size, device)

            optimizer.zero_grad()
            if inputs.ndimension() == 2:  # If the input is [batch_size, seq_len]
                inputs = inputs.unsqueeze(-1)

            _, outputs = model(inputs, hidden, noise=True)
            
            if targets.ndimension() == 2:  # If the target is [batch_size, seq_len]
                targets = targets.unsqueeze(-1)  # Now the shape is [batch_size, seq_len, 1]

            if mask_loss:
                loss = masked_loss(outputs, targets, loss_fxn, mask)
            else:
                loss = loss_fxn(outputs, targets)

            loss.backward()

            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(dataloader)
        losses.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in validationloader:
                if mask_loss:
                    val_inputs, val_targets, val_mask = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
                else:
                    val_inputs, val_targets = val_data[0].to(device), val_data[1].to(device)
                val_batch_size = val_inputs.size(0)
                hidden = model.init_hidden(val_batch_size, device)

                if val_inputs.ndimension() == 2:  # If the input is [batch_size, seq_len]
                    val_inputs = val_inputs.unsqueeze(-1)
                    
                _, val_outputs = model(val_inputs, hidden, noise=True)

                if val_targets.ndimension() == 2:
                    val_targets = val_targets.unsqueeze(-1)

                if mask_loss:
                    val_loss += masked_loss(val_outputs, val_targets, loss_fxn, val_mask)
                else:  
                    val_loss += loss_fxn(val_outputs, val_targets).item()

        val_loss /= len(validationloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), savepath + save_name)
            print(f"Model improved and saved at epoch {epoch+1} with val loss {min_loss:.4f}")

        # Early stopping
        if val_loss <= early_stop_loss:
            print("Early stopping threshold reached.")
            break

    return model, val_losses, losses

