import torch 

def train_RNN(model, dataloader, validationloader, optimizer, loss_fxn, config, device, seed):
    
    min_loss = float('inf')
    val_losses = []
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(config['training']['epocs']):
        model.train()
        model.zero_grad()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            hidden = torch.rand(model.hidden_size, generator=generator).to(device)
            outputs = []

            for t in range(inputs.size(1)): #inefficient, equinox way was way faster
                input_t = inputs[:, t, :]
                hidden = model(input_t, hidden)
                outputs.append(model.ho(model.activation(hidden)))

            outputs = torch.stack(outputs, dim=1)
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
                hidden = torch.rand(model.hidden_size, generator=generator).to(device)
                val_outputs = []

                for t in range(val_inputs.size(1)):
                    input_t = val_inputs[:, t, :]
                    hidden = model(input_t, hidden)
                    val_outputs.append(hidden)

                val_outputs = torch.stack(val_outputs, dim=1)
                val_loss += loss_fxn(val_outputs, val_targets).item()

        val_loss /= len(validationloader)
        print(f"Epoch {epoch+1}/{config['training']['epocs']}, Loss: {loss.item()}, Validation Loss: {val_loss}")
        val_losses.append(val_loss)
        # Save the model if validation loss improves
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), str(seed) + '_' + config['training']['save_path'])
            print(f"Model saved with improvement at epoch {epoch+1} with loss {min_loss}.")

        if val_loss <= config['training']['early_stopping_loss']:
            print("Early stopping threshold reached.")
            break

    return model, val_losses



