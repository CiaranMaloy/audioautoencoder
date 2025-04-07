import torch
from tqdm import tqdm
from loss import *

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

    def process(self):
        """Return the stored evaluation results as a Pandas DataFrame."""
        return pd.DataFrame(self.results)