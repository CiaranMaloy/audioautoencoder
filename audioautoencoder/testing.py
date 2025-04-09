import torch
from tqdm import tqdm
from loss import *

# Testing loop
def test_model_gpu(model, test_loader, criterion, scalers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for inputs, targets, _ in progress_bar:

          inputs, targets = inputs.to(device), targets.to(device)

          outputs = model(inputs)
          loss = criterion(outputs, targets)
          progress_bar.set_postfix(loss=f"{loss.item():.4f}")
          test_loss += loss.item()

        test_loss /= len(test_loader)

    return test_loss

# Testing loop
def test_model(model, test_loader, criterion, scalers):
    evaluation = Evaluation(scalers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for inputs, targets, metadata in progress_bar:

          inputs, targets = inputs.to(device), targets.to(device)

          outputs = model(inputs)
          loss = criterion(outputs, targets)
          progress_bar.set_postfix(loss=f"{loss.item():.4f}")
          test_loss += loss.item()

          # evaluation
          evaluation.evaluate(inputs, targets, outputs, metadata)

        test_loss /= len(test_loader)

    return test_loss, evaluation.process()

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torchmetrics.audio import SignalDistortionRatio
import numpy as np

class Evaluation:
    def __init__(self, scalers, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize storage for evaluation metrics."""
        self.results = []
        self.device = torch.device(device)  # Store the device
        self.sdr = SignalDistortionRatio().to(self.device)  # Move SDR metric to the device
        self.scalers = scalers


    def evaluate(self, inputs, targets, outputs, metadata):
        """
        Compute SDR and L1 loss for input vs. target and input vs. output.
        """
        batch_size = inputs.shape[0]
        chunk_length = 44100 * 2

        # I need to inverse the standardisation for this, as it is skewing the results

        # Convert tensors to NumPy
        inputs = inputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        for i in range(batch_size):
            filename = metadata[i]["filename"]
            snr_db = metadata[i]["snr_db"]
            #phase = metadata[i]["phase"]
            lf_shape = metadata[i]["lf_shape"]

            input = inputs[i]
            output = outputs[i]
            target = targets[i]

            in_track = input.copy()
            out_track = output.copy()
            tar_track = target.copy()

            # Inverse standardisation
            #input_temp = input[0]
            #input[0] = self.scalers["input_features_spectrogram"].transform(input_temp.reshape(1, -1)).reshape(input_temp.shape)
            #output_temp = output[0]
            #output[0] = self.scalers["target_features_spectrogram"].transform(output_temp.reshape(1, -1)).reshape(output_temp.shape)
            #target_temp = target[0]
            #target[0] = self.scalers["target_features_spectrogram"].transform(target_temp.reshape(1, -1)).reshape(target_temp.shape)

            #input_chunk = magphase_to_waveform(input[0], input[1], chunk_length)
            #output_chunk = magphase_to_waveform(output[0], input[1], chunk_length)
            #target_chunk = magphase_to_waveform(target[0], input[1], chunk_length)

            # Move tensors to the correct device
            #input_chunk = torch.from_numpy(input_chunk).to(self.device).float()
            #output_chunk = torch.from_numpy(output_chunk).to(self.device).float()
            #target_chunk = torch.from_numpy(target_chunk).to(self.device).float()

            input = torch.from_numpy(input).to(self.device).float()
            output = torch.from_numpy(output).to(self.device).float()
            target = torch.from_numpy(target).to(self.device).float()

            # Compute SDR (using torchaudio)
            #sdr_invstar = self.sdr(input_chunk, target_chunk).item()
            #sdr_outvstar = self.sdr(output_chunk, target_chunk).item()

            # Compute L1 loss
            l1_invstar = F.l1_loss(input[0:4, :, :], target[0:4, :, :]).item()
            l1_outvstar = F.l1_loss(output[0:4, :, :], target[0:4, :, :]).item()

            l1_invstar_4k = F.l1_loss(input[1:2, :, :], target[1:2, :, :]).item()
            l1_outvstar_4k = F.l1_loss(output[1:2, :, :], target[1:2, :, :]).item()

            l1_invstar_full = F.l1_loss(input[0:1, :, :], target[0:1, :, :]).item()
            l1_outvstar_full = F.l1_loss(output[0:1, :, :], target[0:1, :, :]).item()

            # Store results
            self.results.append({
                "instance": len(self.results),
                #"sdr_invstar": sdr_invstar,
                #"sdr_outvstar": sdr_outvstar,
                "l1_invstar": l1_invstar,
                "l1_outvstar": l1_outvstar,
                "l1_invstar_4k": l1_invstar_4k,
                "l1_outvstar_4k": l1_outvstar_4k,
                "l1_invstar_full": l1_invstar_full,
                "l1_outvstar_full": l1_outvstar_full,
                "filename": filename,
                "snr_db": snr_db,
                "in_track":in_track,
                "out_track":out_track,
                "tar_track":tar_track,
                "metadata": metadata[i],
            })

    def evaluate_dataset(self, inputs, targets, metadata):
        """
        Compute SDR and L1 loss for input vs. target and input vs. output.
        """
        batch_size = inputs.shape[0]

        # Convert tensors to NumPy
        inputs = inputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        for i in range(batch_size):
            filename = metadata[i]["filename"]
            snr_db = metadata[i]["snr_db"]

            input = inputs[i]
            target = targets[i]

            # Basic stats
            in_max = np.max(input)
            tar_max = np.max(target)

            in_min = np.min(input)
            tar_min = np.min(target)

            in_mean = np.mean(input)
            tar_mean = np.mean(target)

            in_std = np.std(input)
            tar_std = np.std(target)

            in_var = np.var(input)
            tar_var = np.var(target)

            in_median = np.median(input)
            tar_median = np.median(target)

            # Additional insights
            in_range = in_max - in_min
            tar_range = tar_max - tar_min

            in_iqr = np.percentile(input, 75) - np.percentile(input, 25)
            tar_iqr = np.percentile(target, 75) - np.percentile(target, 25)

            in_skew = (np.mean((input - in_mean)**3)) / (in_std**3 + 1e-8)
            tar_skew = (np.mean((target - tar_mean)**3)) / (tar_std**3 + 1e-8)

            in_kurtosis = (np.mean((input - in_mean)**4)) / (in_std**4 + 1e-8)
            tar_kurtosis = (np.mean((target - tar_mean)**4)) / (tar_std**4 + 1e-8)

            # Sparsity
            in_sparsity = np.sum(input == 0) / input.size
            tar_sparsity = np.sum(target == 0) / target.size

            # Energy
            in_energy = np.sum(np.square(input))
            tar_energy = np.sum(np.square(target))

            # Entropy
            def entropy(x, bins=100):
                hist, _ = np.histogram(x, bins=bins, density=True)
                hist = hist[hist > 0]
                return -np.sum(hist * np.log(hist))

            in_entropy = entropy(input)
            tar_entropy = entropy(target)

            # Move back to torch
            input = torch.from_numpy(input).to(self.device).float()
            target = torch.from_numpy(target).to(self.device).float()

            # Compute L1 loss
            l1_invstar = F.l1_loss(input, target).item()

            # Store results
            self.results.append({
                "instance": len(self.results),
                "l1_invstar": l1_invstar,
                "filename": filename,
                "snr_db": snr_db,
                "metadata": metadata[i],

                # Input metrics
                "in_max": in_max,
                "in_min": in_min,
                "in_mean": in_mean,
                "in_std": in_std,
                "in_var": in_var,
                "in_median": in_median,
                "in_range": in_range,
                "in_iqr": in_iqr,
                "in_skew": in_skew,
                "in_kurtosis": in_kurtosis,
                "in_sparsity": in_sparsity,
                "in_energy": in_energy,
                "in_entropy": in_entropy,

                # Target metrics
                "tar_max": tar_max,
                "tar_min": tar_min,
                "tar_mean": tar_mean,
                "tar_std": tar_std,
                "tar_var": tar_var,
                "tar_median": tar_median,
                "tar_range": tar_range,
                "tar_iqr": tar_iqr,
                "tar_skew": tar_skew,
                "tar_kurtosis": tar_kurtosis,
                "tar_sparsity": tar_sparsity,
                "tar_energy": tar_energy,
                "tar_entropy": tar_entropy,
            })


    def process(self):
        """Return the stored evaluation results as a Pandas DataFrame."""
        return pd.DataFrame(self.results)
    
def test_dataset(test_loader, scalers):
    evaluation = Evaluation(scalers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for inputs, targets, metadata in progress_bar:

          inputs, targets = inputs.to(device), targets.to(device)

          # evaluation
          evaluation.evaluate_dataset(inputs, targets, metadata)

    return evaluation.process()
