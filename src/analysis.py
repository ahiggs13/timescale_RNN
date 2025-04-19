import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_example(model, batch, device, generator, savepath):
    """
    Plot an example input-output pair from the dataset
    """

    model.eval()
    with torch.no_grad():
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        hidden = torch.rand((batch_size, model.hidden_size), generator=generator).to(device)
        outputs = []

        for t in range(inputs.size(1)):
            input_t = inputs[:, t].unsqueeze(1)  # slow, would be better if all at once
            hidden, output = model(input_t, hidden, noise=False)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.squeeze(-1).cpu().numpy()
        targets = targets.squeeze(-1).cpu().numpy()
        inputs = inputs.squeeze(-1).cpu().numpy()

        for i, input in enumerate(inputs):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(outputs[i], label='Predictions')
            ax.plot(targets[i], label="Targets")
            ax.plot(inputs[i], label='Inputs')
            ax.legend()

            plt.savefig(os.path.join(savepath, f'output_{i}.png'))
            plt.close(fig)
