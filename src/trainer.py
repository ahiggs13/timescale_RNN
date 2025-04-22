import torch 

def train_RNN(model, dataloader, validationloader, optimizer, loss_fxn, config, device, generator, savepath):
    
    min_loss = float('inf')
    val_losses = []

    for epoch in range(config['training']['epocs']):
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size).to(device)  # Initialize hidden state
            outputs = []

            optimizer.zero_grad()
            # Process inputs one timestep at a time to avoid large batch issues
            for t in range(inputs.size(1)): # slow, would be better if all at once
                input_t = inputs[:, t].unsqueeze(1)  
                hidden, output = model(input_t, hidden, noise=True)
                outputs.append(output)

            outputs = torch.cat(outputs, dim=1)  # Concatenate outputs along the time dimension
            loss = loss_fxn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(dataloader)
            
        # Validation step 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in validationloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                batch_size = val_inputs.size(0)
                hidden = model.init_hidden(batch_size).to(device)  # Initialize hidden state
                val_outputs = []

                for t in range(val_inputs.size(1)):
                    input_t = val_inputs[:, t].unsqueeze(1)
                    hidden, output = model(input_t, hidden, noise=True)
                    val_outputs.append(output)

                val_outputs = torch.cat(val_outputs, dim=1)
                val_loss += loss_fxn(val_outputs, val_targets).item()

        val_loss /= len(validationloader)
        print(f"Epoch {epoch+1}/{config['training']['epocs']}, Loss: {total_loss}, Validation Loss: {val_loss}")
        val_losses.append(val_loss)

        # Save the model if validation loss improves
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), savepath + config['training']['save_path'])
            print(f"Model saved with improvement at epoch {epoch+1} with loss {min_loss}.")

        if val_loss <= config['training']['early_stopping_loss']:
            print("Early stopping threshold reached.")
            break

    return model, val_losses



