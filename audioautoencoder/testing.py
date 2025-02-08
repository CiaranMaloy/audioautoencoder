import torch
from tqdm import tqdm
from loss import *

# Testing loop
def test_model(model, test_loader, criterion):
    evaluation = Evaluation()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for inputs, targets in progress_bar:

          print(inputs)
          #inputs, targets = inputs.to(device), targets.to(device)

          #outputs = model(inputs)
          #loss = criterion(outputs, targets)
          #progress_bar.set_postfix(loss=f"{loss.item():.4f}")
          #test_loss += loss.item()

          # evaluation
          #evaluation.evaluate(inputs, targets, outputs)

        test_loss /= len(test_loader)

    return test_loss, evaluation.process()

import torch
import torchaudio.functional as F
import torch.nn.functional as nnF
import pandas as pd
from tqdm import tqdm

class Evaluation:
    def __init__(self):
        """Initialize storage for evaluation metrics."""
        self.results = []

    def evaluate(self, inputs, targets, outputs):
        """
        Compute SDR and L1 loss for input vs. target and input vs. output.
        
        Args:
            inputs (torch.Tensor): The noisy input signal.
            targets (torch.Tensor): The clean target signal.
            outputs (torch.Tensor): The denoised output signal.
        """
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            input_signal = inputs[i].detach().cpu()
            target_signal = targets[i].detach().cpu()
            output_signal = outputs[i].detach().cpu()

            # Compute SDR (using torchaudio)
            sdr_invstar = F.sdr(input_signal, target_signal).item()
            sdr_invsout = F.sdr(input_signal, output_signal).item()

            # Compute L1 loss
            l1_invstar = nnF.l1_loss(input_signal, target_signal).item()
            l1_invsout = nnF.l1_loss(input_signal, output_signal).item()

            # Store results
            self.results.append({
                "instance": len(self.results),
                "sdr_invstar": sdr_invstar,
                "sdr_invsout": sdr_invsout,
                "l1_invstar": l1_invstar,
                "l1_invsout": l1_invsout
            })

    def process(self):
        """Return the stored evaluation results as a Pandas DataFrame."""
        return pd.DataFrame(self.results)


def compute_sdr_torchaudio(reference, estimated):
    """
    Compute SDR using torchaudio's built-in function.

    Parameters:
        reference (numpy.ndarray or torch.Tensor): Ground truth source (shape: [num_sources, time])
        estimated (numpy.ndarray or torch.Tensor): Estimated separated source (shape: [num_sources, time])

    Returns:
        torch.Tensor: SDR values per source
    """
    if not isinstance(reference, torch.Tensor):
        reference = torch.tensor(reference, dtype=torch.float32)
    if not isinstance(estimated, torch.Tensor):
        estimated = torch.tensor(estimated, dtype=torch.float32)

    sdr = F.sdr(estimated, reference)
    return sdr

# Testing loop
def test_examples(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            #input_features = input_features.to(device)
            clean_imgs = clean_imgs.to(device)
            outputs = model(noisy_imgs)
            break  # Display only the first batch

    return noisy_imgs.cpu(), outputs.cpu(), clean_imgs.cpu()