import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, output_time_length=86, channels=2):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file once
        self.input_dataset = self.h5_file["input_images"]
        self.target_dataset = self.h5_file["target_images"]

        # Store dataset shapes for length checking
        self.input_shape = self.input_dataset.shape
        self.target_shape = self.target_dataset.shape

        print("Input dataset shape:", self.input_shape)
        print("Target dataset shape:", self.target_shape)

        # Ensure the input and target datasets are aligned
        assert self.input_shape[0] == self.target_shape[0], "Mismatch in input and target dataset sizes"

    def __len__(self):
        return self.input_shape[0]

    def __getitem__(self, idx):
        try:

            input_image = torch.tensor(self.input_dataset[idx, :, :, :self.output_time_length], dtype=torch.float32)

            if self.channels == 1:
                target_image = torch.tensor(self.target_dataset[idx, 0:1, :, :self.output_time_length], dtype=torch.float32)
            else:
                target_image = torch.tensor(self.target_dataset[idx, :, :, :self.output_time_length], dtype=torch.float32)
              
            return input_image, target_image

        except Exception as e:
          # Log the error and index
          print('\n')
          print(f"Error loading sample {idx}: {e}")

          # Return placeholder tensors
          input_placeholder = torch.zeros(self.input_shape[1:], dtype=torch.float32)
          target_placeholder = torch.zeros(self.target_shape[1:], dtype=torch.float32)
          return input_placeholder, target_placeholder

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()


import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

class HDF5Dataset_metadata(Dataset):
    def __init__(self, h5_file_path, output_time_length=86, channels=2):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file once

        self.input_dataset = self.h5_file["input_images"]
        self.target_dataset = self.h5_file["target_images"]
        self.filename_dataset = self.h5_file["filenames"]
        self.snr_db_dataset = self.h5_file["snr_db"]

        # Store dataset shapes for length checking
        self.input_shape = self.input_dataset.shape
        self.target_shape = self.target_dataset.shape

        print("Input dataset shape:", self.input_shape)
        print("Target dataset shape:", self.target_shape)

        # Ensure input and target datasets are aligned
        assert self.input_shape[0] == self.target_shape[0], "Mismatch in input and target dataset sizes"

    def __len__(self):
        return self.input_shape[0]

    def __getitem__(self, idx):
        try:
            input_image = torch.tensor(self.input_dataset[idx, :, :, :self.output_time_length], dtype=torch.float32)

            if self.channels == 1:
                target_image = torch.tensor(self.target_dataset[idx, 0:1, :, :self.output_time_length], dtype=torch.float32)
            else:
                target_image = torch.tensor(self.target_dataset[idx, :, :, :self.output_time_length], dtype=torch.float32)

            # Extract filename correctly
            filename = self.filename_dataset[idx]
            if isinstance(filename, bytes):  # Check if it's a bytes object
                filename = filename.decode('utf-8')  # Convert to a string

            # Extract metadata
            metadata = {
                "filename": filename,
                "snr_db": self.snr_db_dataset[idx].item() # Convert to Python float
            }

            return input_image, target_image, metadata

        except Exception as e:
            print(f"\nError loading sample {idx}: {e}")

            # Return placeholder tensors and empty metadata
            input_placeholder = torch.zeros((self.input_shape[1], self.input_shape[2], self.output_time_length), dtype=torch.float32)
            target_placeholder = torch.zeros((self.target_shape[1], self.target_shape[2], self.output_time_length), dtype=torch.float32)
            metadata_placeholder = {"filename": [""], "snr_db": float("nan")}

            return input_placeholder, target_placeholder, metadata_placeholder

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

import torch.nn.functional as F
class HDF5Dataset_features(Dataset):
    def __init__(self, h5_file_path, scalers, output_time_length=86, channels=2):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.scalers = scalers
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file once

        # this is where I need to put the scalars for each batch
        # Load input features
        #input_phase_dataset = self.h5_file["input_features_phase"]
        #input_spectrogram_dataset = self.h5_file["input_features_spectrogram"]
        #input_edges_dataset = self.h5_file["input_features_edges"]
        #input_mfcc_dataset = self.h5_file["input_features_mfccs"]
        #input_mfcc_delta_dataset = self.h5_file["input_features_mfcc_delta"]
        #input_mfcc_delta2_dataset = self.h5_file["input_features_mfcc_delta2"]

        # Load target
        #target_spectrogram = self.h5_file["target_features_spectrogram"]

        # metadata
        #self.filename_dataset = self.h5_file["filenames"]
        #self.snr_db_dataset = self.h5_file["snr_db"]

        # Store dataset shapes for length checking
        #self.input_shape = self.input_dataset.shape
        #self.target_shape = self.target_dataset.shape

        print("Dataset size:", self.h5_file["snr_db"].shape[0])

    def __len__(self):
        return self.h5_file["snr_db"].shape[0]
    
    def resample_feature(self, feature, target_shape):
        """Resamples a 2D numpy feature array to match target shape using torch.nn.functional.interpolate."""
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        target_size = (target_shape[0], target_shape[1])  # (new_H, new_W)
        
        resized_feature = F.interpolate(feature_tensor, size=target_size, mode="bilinear", align_corners=False)
        return resized_feature.squeeze(0).squeeze(0).numpy()  # Remove batch/channel dim and return as numpy

    def __getitem__(self, idx):
        #try:
            # normalise dataset (maybe try and keep in torch language)

            # this is where to combine the features into a 5 channel image after each image has been normalised
            
        # Load input features
        input_phase = self.h5_file["input_features_phase"][idx]
        input_spectrogram = self.h5_file["input_features_spectrogram"][idx]
        input_edges = self.h5_file["input_features_edges"][idx]
        input_cepstrum = self.h5_file["input_features_cepstrum"][idx]
        input_cepstrum_edges= self.h5_file["input_features_cepstrum_edges"][idx]

        # Define target shape (use spectrogram shape as reference)
        target_shape = input_spectrogram.shape

        # Load target
        target_spectrogram = self.h5_file["target_features_spectrogram"][idx]

        # Apply scalers
        input_phase = self.scalers["input_features_phase"].transform(input_phase.reshape(1, -1)).reshape(input_phase.shape)
        input_spectrogram = self.scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)
        input_edges = self.scalers["input_features_edges"].transform(input_edges.reshape(1, -1)).reshape(input_edges.shape)
        input_cepstrum = self.scalers["input_features_cepstrum"].transform(input_cepstrum.reshape(1, -1)).reshape(input_cepstrum.shape)
        input_cepstrum_edges = self.scalers["input_features_cepstrum_edges"].transform(input_cepstrum_edges.reshape(1, -1)).reshape(input_cepstrum_edges.shape)

        target_spectrogram = self.scalers["target_features_spectrogram"].transform(target_spectrogram.reshape(1, -1)).reshape(target_spectrogram.shape)

        # resample mfcc featues so theyre the same shape as the spectrogram and phase features
        # Define frequency bins
        sampling_rate = 44100  # 44.1 kHz audio
        n_fft = 2048  # Adjust this for better resolution
        freqs = np.linspace(0, sampling_rate / 2, n_fft // 2 + 1)  # STFT frequency bins

        # Find indices corresponding to 0–4000 Hz
        min_freq, max_freq = 0, 4000
        freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
        # Resample MFCC features
        input_cepstrum = self.resample_feature(input_cepstrum, target_shape)
        input_cepstrum_edges = self.resample_feature(input_cepstrum[:500, :], target_shape)
        #input_mfcc_delta2 = self.resample_feature(input_mfcc_delta2, target_shape)
        input_spectrogram_lf = self.resample_feature(input_spectrogram[freq_indices, :], target_shape)

        # Convert to tensors - input_phase, is missing,..... it's too confusing
        inputs = torch.tensor(np.stack([
            input_spectrogram, input_spectrogram_lf, input_edges,
            input_cepstrum, input_cepstrum_edges
        ], axis=0), dtype=torch.float32)  # Shape: (6, H, W)

        # Output:
        target_spectrogram_lf = self.resample_feature(target_spectrogram[freq_indices, :], target_shape)
        target = torch.tensor(np.stack([
            target_spectrogram, target_spectrogram_lf
        ], axis=0), dtype=torch.float32) 

        # reformat to between 0 and 1
        a = 3
        inputs = (inputs/a) + 0.5
        target = (target/a) + 0.5

        # Extract filename correctly
        filename = self.h5_file["filenames"][idx]
        if isinstance(filename, bytes):  # Check if it's a bytes object
            filename = filename.decode('utf-8')  # Convert to a string

        # Extract metadata
        metadata = {
            "filename": filename,
            "snr_db": self.h5_file["snr_db"][idx].item(), # Convert to Python float
            "phase": input_phase,
            "lf_shape": input_spectrogram[freq_indices, :].shape,
            "freq_indicies": freq_indices
        }

        return inputs, target, metadata

        #except Exception as e:
        #    print(f"\nError loading sample {idx}: {e}")

            # Return placeholder tensors and empty metadata
        #    input_placeholder = torch.zeros((self.input_shape[1], self.input_shape[2], self.output_time_length), dtype=torch.float32)
        #    target_placeholder = torch.zeros((self.target_shape[1], self.target_shape[2], self.output_time_length), dtype=torch.float32)
        #    metadata_placeholder = {"filename": [""], "snr_db": float("nan")}

        #    return input_placeholder, target_placeholder, metadata_placeholder

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

import torch.nn.functional as F
class HDF5Dataset_bandchannels(Dataset):
    def __init__(self, h5_file_path, scalers, output_time_length=86, channels=2):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.scalers = scalers
        self.a = 2

        #print("Dataset size:", self.h5_file["snr_db"].shape[0])

    def __len__(self):
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file once
        return self.h5_file["snr_db"].shape[0]
    
    def resample_feature(self, feature, target_shape):
        """Resamples a 2D numpy feature array to match target shape using torch.nn.functional.interpolate."""
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        target_size = (target_shape[0], target_shape[1])  # (new_H, new_W)
        
        resized_feature = F.interpolate(feature_tensor, size=target_size, mode="bilinear", align_corners=False)
        return resized_feature.squeeze(0).squeeze(0).numpy()  # Remove batch/channel dim and return as numpy

    def downsample_H_by_factor(self, inputs, scale_factor):
        B, H, W = inputs.shape
        new_H = int(H // scale_factor)  # Compute new height

        # Unsqueeze a channel dimension -> (B, 1, H, W)
        inputs = inputs.unsqueeze(1)

        # Resize only height using bilinear interpolation
        resampled = F.interpolate(inputs, size=(new_H, W), mode="bilinear", align_corners=False)

        return resampled.squeeze(1)  # Remove the channel dimension

    def __getitem__(self, idx):
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file per worker
        #try:
            # normalise dataset (maybe try and keep in torch language)

            # this is where to combine the features into a 5 channel image after each image has been normalised
            
        # Load input features
        input_phase = self.h5_file["input_features_phase"][idx]
        input_spectrogram = self.h5_file["input_features_spectrogram"][idx]
        input_edges = self.h5_file["input_features_edges"][idx]
        input_cepstrum = self.h5_file["input_features_cepstrum"][idx]
        #input_cepstrum_edges= self.h5_file["input_features_cepstrum_edges"][idx]

        # Define target shape (use spectrogram shape as reference)
        target_shape = input_spectrogram.shape

        # Load target
        target_spectrogram = self.h5_file["target_features_spectrogram"][idx]

        # Apply scalers
        #input_phase = self.scalers["input_features_phase"].transform(input_phase.reshape(1, -1)).reshape(input_phase.shape)
        input_spectrogram = self.scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)
        input_edges = self.scalers["input_features_edges"].transform(input_edges.reshape(1, -1)).reshape(input_edges.shape)
        input_cepstrum = self.scalers["input_features_cepstrum"].transform(input_cepstrum.reshape(1, -1)).reshape(input_cepstrum.shape)
        #input_cepstrum_edges = self.scalers["input_features_cepstrum_edges"].transform(input_cepstrum_edges.reshape(1, -1)).reshape(input_cepstrum_edges.shape)

        target_spectrogram = self.scalers["target_features_spectrogram"].transform(target_spectrogram.reshape(1, -1)).reshape(target_spectrogram.shape)

        # resample mfcc featues so theyre the same shape as the spectrogram and phase features
        # Define frequency bins
        sampling_rate = 44100  # 44.1 kHz audio
        n_fft = 2048  # Adjust this for better resolution
        freqs = np.linspace(0, sampling_rate / 2, n_fft // 2 + 1)  # STFT frequency bins

        # Find indices corresponding to 0–4000 Hz
        # updated bandchannels will be 5000, 1250, 500
        min_freq, hf, mf, lf = 0, 5000, 1250, 500
        freq_indices_hf = np.where((freqs >= min_freq) & (freqs <= hf))[0]
        freq_indices_mf = np.where((freqs >= min_freq) & (freqs <= mf))[0]
        freq_indices_lf = np.where((freqs >= min_freq) & (freqs <= lf))[0]
        # input spectrogram
        input_spectrogram_hf = self.resample_feature(input_spectrogram[freq_indices_hf, :], target_shape)
        input_spectrogram_mf = self.resample_feature(input_spectrogram[freq_indices_mf, :], target_shape)
        input_spectrogram_lf = self.resample_feature(input_spectrogram[freq_indices_lf, :], target_shape)
        # edges
        #input_edges_hf = self.resample_feature(input_edges[freq_indices_hf, :], target_shape)
        input_edges_mf = self.resample_feature(input_edges[freq_indices_mf, :], target_shape)
        #input_edges_lf = self.resample_feature(input_edges[freq_indices_lf, :], target_shape)
        # Resample MFCC features
        input_cepstrum = self.resample_feature(input_cepstrum, target_shape)
        input_cepstrum_hf = self.resample_feature(input_cepstrum[freq_indices_hf, :], target_shape)
        input_cepstrum_mf = self.resample_feature(input_cepstrum[freq_indices_mf, :], target_shape)
        input_cepstrum_lf = self.resample_feature(input_cepstrum[freq_indices_lf, :], target_shape)

        # target
        target_spectrogram_hf = self.resample_feature(target_spectrogram[freq_indices_hf, :], target_shape)
        target_spectrogram_mf = self.resample_feature(target_spectrogram[freq_indices_mf, :], target_shape)
        target_spectrogram_lf = self.resample_feature(target_spectrogram[freq_indices_lf, :], target_shape)


        # Convert to tensors - input_phase, is missing,..... it's too confusing
        inputs = torch.tensor(np.stack([
            input_spectrogram, input_spectrogram_hf, input_spectrogram_mf, input_spectrogram_lf,
            input_cepstrum, input_cepstrum_hf, input_cepstrum_mf, input_cepstrum_lf,
            input_edges_mf
        ], axis=0), dtype=torch.float32)  # Shape: (6, H, W)

        # Output:
        target = torch.tensor(np.stack([
            target_spectrogram, target_spectrogram_hf, target_spectrogram_mf, target_spectrogram_lf
        ], axis=0), dtype=torch.float32) 

        # reformat to between 0 and 1
        #inputs = (inputs/a) + 0.5
        #target = (target/a) + 0.5

        # reformat to between 0 and 1
        inputs = torch.clamp((inputs/self.a) + 0.5, min=0)
        target = torch.clamp((target/self.a) + 0.5, min=0)

        inputs = self.downsample_H_by_factor(inputs, 4)
        target = self.downsample_H_by_factor(target, 4)

        # Extract filename correctly
        filename = self.h5_file["filenames"][idx]
        if isinstance(filename, bytes):  # Check if it's a bytes object
            filename = filename.decode('utf-8')  # Convert to a string

        # Extract metadata
        metadata = {
            "filename": filename,
            "snr_db": self.h5_file["snr_db"][idx].item(), # Convert to Python float
            "phase": input_phase,
            "hf_shape": input_spectrogram[freq_indices_hf, :].shape,
            "mf_shape": input_spectrogram[freq_indices_mf, :].shape,
            "lf_shape": input_spectrogram[freq_indices_lf, :].shape,
            "freq_indices_hf": freq_indices_hf,
            "freq_indices_mf": freq_indices_mf,
            "freq_indices_lf": freq_indices_lf
        }

        return inputs, target, metadata

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

import torch.nn.functional as F
class HDF5Dataset_bandchannels_no_features(Dataset):
    def __init__(self, h5_file_path, scalers, output_time_length=86, channels=2):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.scalers = scalers
        self.a = 2

        #print("Dataset size:", self.h5_file["snr_db"].shape[0])

    def __len__(self):
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file once
        return self.h5_file["snr_db"].shape[0]
    
    def resample_feature(self, feature, target_shape):
        """Resamples a 2D numpy feature array to match target shape using torch.nn.functional.interpolate."""
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        target_size = (target_shape[0], target_shape[1])  # (new_H, new_W)
        
        resized_feature = F.interpolate(feature_tensor, size=target_size, mode="bilinear", align_corners=False)
        return resized_feature.squeeze(0).squeeze(0).numpy()  # Remove batch/channel dim and return as numpy

    def downsample_H_by_factor(self, inputs, scale_factor):
        B, H, W = inputs.shape
        new_H = int(H // scale_factor)  # Compute new height

        # Unsqueeze a channel dimension -> (B, 1, H, W)
        inputs = inputs.unsqueeze(1)

        # Resize only height using bilinear interpolation
        resampled = F.interpolate(inputs, size=(new_H, W), mode="bilinear", align_corners=False)

        return resampled.squeeze(1)  # Remove the channel dimension

    def __getitem__(self, idx):
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file per worker
        #try:
            # normalise dataset (maybe try and keep in torch language)

            # this is where to combine the features into a 5 channel image after each image has been normalised
            
        # Load input features
        #input_phase = self.h5_file["input_features_phase"][idx]
        input_spectrogram = self.h5_file["input_features_spectrogram"][idx]
        #input_edges = self.h5_file["input_features_edges"][idx]
        #input_cepstrum = self.h5_file["input_features_cepstrum"][idx]
        #input_cepstrum_edges= self.h5_file["input_features_cepstrum_edges"][idx]

        # Define target shape (use spectrogram shape as reference)
        target_shape = input_spectrogram.shape

        # Load target
        target_spectrogram = self.h5_file["target_features_spectrogram"][idx]

        # Apply scalers
        #input_phase = self.scalers["input_features_phase"].transform(input_phase.reshape(1, -1)).reshape(input_phase.shape)
        input_spectrogram = self.scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)
        #input_edges = self.scalers["input_features_edges"].transform(input_edges.reshape(1, -1)).reshape(input_edges.shape)
        #input_cepstrum = self.scalers["input_features_cepstrum"].transform(input_cepstrum.reshape(1, -1)).reshape(input_cepstrum.shape)
        #input_cepstrum_edges = self.scalers["input_features_cepstrum_edges"].transform(input_cepstrum_edges.reshape(1, -1)).reshape(input_cepstrum_edges.shape)

        target_spectrogram = self.scalers["target_features_spectrogram"].transform(target_spectrogram.reshape(1, -1)).reshape(target_spectrogram.shape)

        # resample mfcc featues so theyre the same shape as the spectrogram and phase features
        # Define frequency bins
        sampling_rate = 44100  # 44.1 kHz audio
        n_fft = 2048  # Adjust this for better resolution
        freqs = np.linspace(0, sampling_rate / 2, n_fft // 2 + 1)  # STFT frequency bins

        # Find indices corresponding to 0–4000 Hz
        # updated bandchannels will be 5000, 1250, 500
        min_freq, hf, mf, lf = 0, 5000, 1250, 500
        freq_indices_hf = np.where((freqs >= min_freq) & (freqs <= hf))[0]
        freq_indices_mf = np.where((freqs >= min_freq) & (freqs <= mf))[0]
        freq_indices_lf = np.where((freqs >= min_freq) & (freqs <= lf))[0]
        # input spectrogram
        input_spectrogram_hf = self.resample_feature(input_spectrogram[freq_indices_hf, :], target_shape)
        input_spectrogram_mf = self.resample_feature(input_spectrogram[freq_indices_mf, :], target_shape)
        input_spectrogram_lf = self.resample_feature(input_spectrogram[freq_indices_lf, :], target_shape)
        # edges
        #input_edges_hf = self.resample_feature(input_edges[freq_indices_hf, :], target_shape)
        #input_edges_mf = self.resample_feature(input_edges[freq_indices_mf, :], target_shape)
        #input_edges_lf = self.resample_feature(input_edges[freq_indices_lf, :], target_shape)
        # Resample MFCC features
        #input_cepstrum = self.resample_feature(input_cepstrum, target_shape)
        #input_cepstrum_hf = self.resample_feature(input_cepstrum[freq_indices_hf, :], target_shape)
        #input_cepstrum_mf = self.resample_feature(input_cepstrum[freq_indices_mf, :], target_shape)
        #input_cepstrum_lf = self.resample_feature(input_cepstrum[freq_indices_lf, :], target_shape)

        # target
        target_spectrogram_hf = self.resample_feature(target_spectrogram[freq_indices_hf, :], target_shape)
        target_spectrogram_mf = self.resample_feature(target_spectrogram[freq_indices_mf, :], target_shape)
        target_spectrogram_lf = self.resample_feature(target_spectrogram[freq_indices_lf, :], target_shape)


        # Convert to tensors - input_phase, is missing,..... it's too confusing
        inputs = torch.tensor(np.stack([
            input_spectrogram, input_spectrogram_hf, input_spectrogram_mf, input_spectrogram_lf,
        ], axis=0), dtype=torch.float32)  # Shape: (6, H, W)

        # Output:
        target = torch.tensor(np.stack([
            target_spectrogram, target_spectrogram_hf, target_spectrogram_mf, target_spectrogram_lf
        ], axis=0), dtype=torch.float32) 

        # reformat to between 0 and 1
        #inputs = (inputs/a) + 0.5
        #target = (target/a) + 0.5

        # reformat to between 0 and 1
        inputs = torch.clamp((inputs/self.a) + 0.5, min=0)
        target = torch.clamp((target/self.a) + 0.5, min=0)

        inputs = self.downsample_H_by_factor(inputs, 4)
        target = self.downsample_H_by_factor(target, 4)

        # Extract filename correctly
        filename = self.h5_file["filenames"][idx]
        if isinstance(filename, bytes):  # Check if it's a bytes object
            filename = filename.decode('utf-8')  # Convert to a string

        # Extract metadata
        metadata = {
            "filename": filename,
            "snr_db": self.h5_file["snr_db"][idx].item(), # Convert to Python float
            #"phase": input_phase,
            "hf_shape": input_spectrogram[freq_indices_hf, :].shape,
            "mf_shape": input_spectrogram[freq_indices_mf, :].shape,
            "lf_shape": input_spectrogram[freq_indices_lf, :].shape,
            "freq_indices_hf": freq_indices_hf,
            "freq_indices_mf": freq_indices_mf,
            "freq_indices_lf": freq_indices_lf
        }

        return inputs, target, metadata

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

class HDF5Dataset_bandchannels_diffusion(Dataset):
    def __init__(self, h5_file_path, scalers, output_time_length=86, channels=2):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.scalers = scalers
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file once

        self.a = 2

        print("Dataset size:", self.h5_file["snr_db"].shape[0])

    def __len__(self):
        return self.h5_file["snr_db"].shape[0]
    
    def resample_feature(self, feature, target_shape):
        """Resamples a 2D numpy feature array to match target shape using torch.nn.functional.interpolate."""
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        target_size = (target_shape[0], target_shape[1])  # (new_H, new_W)
        
        resized_feature = F.interpolate(feature_tensor, size=target_size, mode="bilinear", align_corners=False)
        return resized_feature.squeeze(0).squeeze(0).numpy()  # Remove batch/channel dim and return as numpy

    def downsample_H_by_factor(self, inputs, scale_factor):
        B, H, W = inputs.shape
        new_H = int(H // scale_factor)  # Compute new height

        # Unsqueeze a channel dimension -> (B, 1, H, W)
        inputs = inputs.unsqueeze(1)

        # Resize only height using bilinear interpolation
        resampled = F.interpolate(inputs, size=(new_H, W), mode="bilinear", align_corners=False)

        return resampled.squeeze(1)  # Remove the channel dimension

    def db_to_amplitude(self, dB, ref=1.0):
        """
        Converts decibels (dB) back to amplitude in PyTorch.

        Args:
            dB (torch.Tensor): dB-scaled spectrogram.
            ref (float, optional): Reference value (default is 1.0).

        Returns:
            torch.Tensor: Amplitude spectrogram.
        """
        return ref * (10.0 ** (dB / 20.0))

    def amplitude_to_db_pytorch(self, A, ref=1.0, top_db=None):
        """
        Converts amplitude to decibels (dB) in PyTorch, similar to librosa.amplitude_to_db.
        
        Args:
            A (torch.Tensor): Amplitude spectrogram.
            ref (float, optional): Reference value. Default is max of A.
            top_db (float, optional): Dynamic range threshold. Default is 80 dB.

        Returns:
            torch.Tensor: dB-scaled spectrogram.
        """
        A = torch.abs(A)  # Ensure positive values
        if ref is None:
            ref = torch.max(A)  # Normalize to max amplitude

        # Convert to dB scale
        log_A = 20.0 * torch.log10(torch.clamp(A / ref, min=1e-10))

        # Apply dynamic range compression (clip to top_db)
        if top_db is not None:
            log_A = torch.clamp(log_A, min=log_A.max() - top_db)

        return log_A

    def amplitude_to_db_numpy(self, A, ref=1.0, top_db=None):
        """
        Converts amplitude to decibels (dB) using NumPy, similar to librosa.amplitude_to_db.

        Args:
            A (np.ndarray): Amplitude spectrogram.
            ref (float, optional): Reference value. Default is max of A.
            top_db (float, optional): Dynamic range threshold.

        Returns:
            np.ndarray: dB-scaled spectrogram.
        """
        A = np.abs(A)  # Ensure positive values
        if ref is None:
            ref = np.max(A)  # Normalize to max amplitude

        # Convert to dB scale
        log_A = 20.0 * np.log10(np.clip(A / ref, a_min=1e-10, a_max=None))

        # Apply dynamic range compression (clip to top_db)
        if top_db is not None:
            log_A = np.clip(log_A, a_min=log_A.max() - top_db, a_max=None)

        return log_A

    def logsubtract(self, a, b):
        a = self.db_to_amplitude(a)
        b = self.db_to_amplitude(b)

        output = a - b
        output = np.clip(output, a_min=1e-12, a_max=None)
        return self.amplitude_to_db_numpy(output)
    
    def __getitem__(self, idx):
        #try:
            # normalise dataset (maybe try and keep in torch language)

            # this is where to combine the features into a 5 channel image after each image has been normalised
            
        # Load input features
        #input_phase = self.h5_file["input_features_phase"][idx]
        #input_spectrogram = self.h5_file["input_features_spectrogram"][idx]
        #input_edges = self.h5_file["input_features_edges"][idx]
        #input_cepstrum = self.h5_file["input_features_cepstrum"][idx]
        #input_cepstrum_edges= self.h5_file["input_features_cepstrum_edges"][idx]

        # Load target
        target_spectrogram = self.h5_file["target_features_spectrogram"][idx]
        #target_spectrogram = self.logsubtract(input_spectrogram, target_spectrogram) 

        # Define target shape (use spectrogram shape as reference)
        target_shape = target_spectrogram.shape

        # Apply scalers
        #input_phase = self.scalers["input_features_phase"].transform(input_phase.reshape(1, -1)).reshape(input_phase.shape)
        #input_spectrogram = self.scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)
        #input_edges = self.scalers["input_features_edges"].transform(input_edges.reshape(1, -1)).reshape(input_edges.shape)
        #input_cepstrum = self.scalers["input_features_cepstrum"].transform(input_cepstrum.reshape(1, -1)).reshape(input_cepstrum.shape)
        #input_cepstrum_edges = self.scalers["input_features_cepstrum_edges"].transform(input_cepstrum_edges.reshape(1, -1)).reshape(input_cepstrum_edges.shape)

        target_spectrogram = self.scalers["target_features_spectrogram"].transform(target_spectrogram.reshape(1, -1)).reshape(target_spectrogram.shape)
        #noise_spectrogram = self.scalers["target_features_spectrogram"].transform(noise_spectrogram.reshape(1, -1)).reshape(target_spectrogram.shape)

        # target
        #target_spectrogram = np.clip(input_spectrogram - target_spectrogram, a_min=0, a_max=None)

        # resample mfcc featues so theyre the same shape as the spectrogram and phase features
        # Define frequency bins
        sampling_rate = 44100  # 44.1 kHz audio
        n_fft = 2048  # Adjust this for better resolution
        freqs = np.linspace(0, sampling_rate / 2, n_fft // 2 + 1)  # STFT frequency bins

        # Find indices corresponding to 0–4000 Hz
        # updated bandchannels will be 5000, 1250, 500
        min_freq, hf, mf, lf = 0, 5000, 1250, 500
        freq_indices_hf = np.where((freqs >= min_freq) & (freqs <= hf))[0]
        freq_indices_mf = np.where((freqs >= min_freq) & (freqs <= mf))[0]
        freq_indices_lf = np.where((freqs >= min_freq) & (freqs <= lf))[0]
        # input spectrogram
        #input_spectrogram_hf = self.resample_feature(input_spectrogram[freq_indices_hf, :], target_shape)
        #input_spectrogram_mf = self.resample_feature(input_spectrogram[freq_indices_mf, :], target_shape)
        #input_spectrogram_lf = self.resample_feature(input_spectrogram[freq_indices_lf, :], target_shape)

        target_spectrogram_hf = self.resample_feature(target_spectrogram[freq_indices_hf, :], target_shape)
        target_spectrogram_mf = self.resample_feature(target_spectrogram[freq_indices_mf, :], target_shape)
        target_spectrogram_lf = self.resample_feature(target_spectrogram[freq_indices_lf, :], target_shape)
        # edges
        #input_edges_hf = self.resample_feature(input_edges[freq_indices_hf, :], target_shape)
        #input_edges_mf = self.resample_feature(input_edges[freq_indices_mf, :], target_shape)
        #input_edges_lf = self.resample_feature(input_edges[freq_indices_lf, :], target_shape)

        # now input indices for 0-1000 and 0-200 to add as channels and as freq_indicies for reconstruction

        # Resample MFCC features
        #input_cepstrum = self.resample_feature(input_cepstrum, target_shape)

        # Convert to tensors - input_phase, is missing,..... it's too confusing
        #inputs = torch.tensor(np.stack([
        #    input_spectrogram, input_spectrogram_hf, input_spectrogram_mf, input_spectrogram_lf,
        #], axis=0), dtype=torch.float32)  # Shape: (6, H, W)
        inputs = None

        # Output:
        target = torch.tensor(np.stack([
            target_spectrogram, target_spectrogram_hf, target_spectrogram_mf, target_spectrogram_lf
        ], axis=0), dtype=torch.float32) 

        # reformat to between 0 and 1
        #inputs = (inputs/self.a) #+ 0.5
        target = (target/self.a) #+ 0.5

        #inputs = self.downsample_H_by_factor(inputs, 4)
        target = self.downsample_H_by_factor(target, 4)

        # Extract filename correctly
        filename = self.h5_file["filenames"][idx]
        if isinstance(filename, bytes):  # Check if it's a bytes object
            filename = filename.decode('utf-8')  # Convert to a string

        # Extract metadata
        metadata = {
            "filename": filename,
            "snr_db": self.h5_file["snr_db"][idx].item(), # Convert to Python float
            #"phase": input_phase,
            "hf_shape": target_spectrogram[freq_indices_hf, :].shape,
            "mf_shape": target_spectrogram[freq_indices_mf, :].shape,
            "lf_shape": target_spectrogram[freq_indices_lf, :].shape,
            "freq_indices_hf": freq_indices_hf,
            "freq_indices_mf": freq_indices_mf,
            "freq_indices_lf": freq_indices_lf
        }

        return inputs, target, metadata

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

class HDF5Dataset_no_features_resampled(Dataset):
    def __init__(self, h5_file_path, scalers, output_time_length=86, channels=2):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.scalers = scalers
        self.a = 2

        #print("Dataset size:", self.h5_file["snr_db"].shape[0])

    def __len__(self):
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file once
        return self.h5_file["snr_db"].shape[0]
    
    def resample_feature(self, feature, target_shape):
        """Resamples a 2D numpy feature array to match target shape using torch.nn.functional.interpolate."""
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        target_size = (target_shape[0], target_shape[1])  # (new_H, new_W)
        
        resized_feature = F.interpolate(feature_tensor, size=target_size, mode="bilinear", align_corners=False)
        return resized_feature.squeeze(0).squeeze(0).numpy()  # Remove batch/channel dim and return as numpy

    def downsample_H_by_factor(self, inputs, scale_factor):
        B, H, W = inputs.shape
        #print(inputs.shape)
        #H, W = inputs.shape
        new_H = int(H // scale_factor)  # Compute new height

        # Unsqueeze a channel dimension -> (B, 1, H, W)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)

        # Resize only height using bilinear interpolation
        resampled = F.interpolate(inputs, size=(new_H, W), mode="bilinear", align_corners=False)

        return resampled.squeeze(1)  # Remove the channel dimension

    def __getitem__(self, idx):
        self.h5_file = h5py.File(self.h5_file_path, "r")  # Open the file per worker
        #try:
            # normalise dataset (maybe try and keep in torch language)

            # this is where to combine the features into a 5 channel image after each image has been normalised
            
        # Load input features
        #input_phase = self.h5_file["input_features_phase"][idx]
        input_spectrogram = self.h5_file["input_features_spectrogram"][idx]
        #input_edges = self.h5_file["input_features_edges"][idx]
        #input_cepstrum = self.h5_file["input_features_cepstrum"][idx]
        #input_cepstrum_edges= self.h5_file["input_features_cepstrum_edges"][idx]

        # Define target shape (use spectrogram shape as reference)
        target_shape = input_spectrogram.shape

        # Load target
        target_spectrogram = self.h5_file["target_features_spectrogram"][idx]

        # Apply scalers
        input_spectrogram = self.scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)
        target_spectrogram = self.scalers["target_features_spectrogram"].transform(target_spectrogram.reshape(1, -1)).reshape(target_spectrogram.shape)

        # resample mfcc featues so theyre the same shape as the spectrogram and phase features
        # Define frequency bins
        sampling_rate = 44100  # 44.1 kHz audio
        n_fft = 2048  # Adjust this for better resolution
        freqs = np.linspace(0, sampling_rate / 2, n_fft // 2 + 1)  # STFT frequency bins

        # Find indices corresponding to 0–4000 Hz
        # updated bandchannels will be 5000, 1250, 500
        octave_base = [0, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        octave_edge_frequencies = [f * np.sqrt(2) for f in octave_base]
        resamle_constant = round((n_fft/2)/(len(octave_edge_frequencies)-1)) * 2

        arrays_input = []
        arrays_target = []
        for i in range(0, len(octave_edge_frequencies)-1):
            freq_indices = np.where((freqs >= octave_edge_frequencies[i]) & (freqs <= octave_edge_frequencies[i+1]))[0]
            print(freq_indices[0], freq_indices[-1])
            arrays_input.append(self.resample_feature(input_spectrogram[freq_indices, :], [target_shape[0], resamle_constant]))
            arrays_target.append(self.resample_feature(target_spectrogram[freq_indices, :], [target_shape[0], resamle_constant]))

        resampled_input = np.vstack(arrays_input)
        resampled_target = np.vstack(arrays_target)

        # input spectrogram
        input_spectrogram = self.resample_feature(resampled_input, target_shape)

        # target
        target_spectrogram = self.resample_feature(resampled_target, target_shape)


        # Convert to tensors - input_phase, is missing,..... it's too confusing
        inputs = torch.tensor([input_spectrogram], dtype=torch.float32)  # Shape: (6, H, W)

        # Output:
        target = torch.tensor([target_spectrogram], dtype=torch.float32) 

        # reformat to between 0 and 1
        #inputs = (inputs/a) + 0.5
        #target = (target/a) + 0.5

        # reformat to between 0 and 1
        inputs = torch.clamp((inputs/self.a) + 0.5, min=0)
        target = torch.clamp((target/self.a) + 0.5, min=0)

        inputs = self.downsample_H_by_factor(inputs, 4)
        target = self.downsample_H_by_factor(target, 4)

        # Extract filename correctly
        filename = self.h5_file["filenames"][idx]
        if isinstance(filename, bytes):  # Check if it's a bytes object
            filename = filename.decode('utf-8')  # Convert to a string

        # Extract metadata
        metadata = {
            "filename": filename,
            "snr_db": self.h5_file["snr_db"][idx].item(), # Convert to Python float
        }

        return inputs, target, metadata

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

import torch
import numpy as np

def custom_collate_fn(batch):
    # Unzip the batch into inputs, targets, and metadata
    inputs, targets, metadata = zip(*batch)
    
    # Convert inputs and targets into tensors (if they are numpy arrays or lists)
    inputs = torch.tensor(np.array(inputs))
    targets = torch.tensor(np.array(targets))
    
    # For metadata, you can keep it as is or process it further if needed
    # Here, assuming metadata is a list of strings (paths)
    
    # Return the batch as a tuple of tensors and metadata
    return inputs, targets, metadata

def custom_collate_fn_diffusion(batch):
    # Unzip the batch into inputs, targets, and metadata
    inputs, targets, metadata = zip(*batch)
    
    # Convert inputs and targets into tensors (if they are numpy arrays or lists)
    #inputs = torch.tensor(np.array(inputs))
    targets = torch.tensor(np.array(targets))
    
    # For metadata, you can keep it as is or process it further if needed
    # Here, assuming metadata is a list of strings (paths)
    
    # Return the batch as a tuple of tensors and metadata
    return inputs, targets, metadata