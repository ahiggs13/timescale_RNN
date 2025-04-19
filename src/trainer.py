import torch 

def train_RNN(model, dataloader, validationloader, optimizer, loss_fxn, config, device, seed, savepath):
    
    min_loss = float('inf')
    val_losses = []
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(config['training']['epocs']):
        model.train()
        model.zero_grad()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            hidden = torch.rand((batch_size, model.hidden_size), generator=generator).to(device)
            outputs = []

            # Process inputs one timestep at a time to avoid large batch issues
            for t in range(inputs.size(1)):
                input_t = inputs[:, t].unsqueeze(1)  # slow, would be better if all at once
                hidden, output = model(input_t, hidden)
                outputs.append(output)

            outputs = torch.cat(outputs, dim=1)  # Concatenate outputs along the time dimension
            loss = loss_fxn(outputs, targets)
            loss.backward()
            optimizer.step()
            

        # Validation step 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in validationloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                batch_size = val_inputs.size(0)
                hidden = torch.rand((batch_size, model.hidden_size), generator=generator).to(device)
                val_outputs = []

                for t in range(val_inputs.size(1)):
                    input_t = val_inputs[:, t].unsqueeze(1)
                    hidden, output = model(input_t, hidden)
                    val_outputs.append(output)

                val_outputs = torch.cat(val_outputs, dim=1)
                val_loss += loss_fxn(val_outputs, val_targets).item()

        val_loss /= len(validationloader)
        print(f"Epoch {epoch+1}/{config['training']['epocs']}, Loss: {loss.item()}, Validation Loss: {val_loss}")
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



