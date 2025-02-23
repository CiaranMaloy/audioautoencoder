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
        input_mfcc = self.h5_file["input_features_mfccs"][idx]
        input_mfcc_delta = self.h5_file["input_features_mfcc_delta"][idx]
        input_mfcc_delta2 = self.h5_file["input_features_mfcc_delta2"][idx]

        # Define target shape (use spectrogram shape as reference)
        target_shape = input_spectrogram.shape

        # Load target
        target_spectrogram = self.h5_file["target_features_spectrogram"][idx]

        # Apply scalers
        input_phase = self.scalers["input_features_phase"].transform(input_phase.reshape(1, -1)).reshape(input_phase.shape)
        input_spectrogram = self.scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)
        input_edges = self.scalers["input_features_edges"].transform(input_edges.reshape(1, -1)).reshape(input_edges.shape)
        input_mfcc = self.scalers["input_features_mfccs"].transform(input_mfcc.reshape(1, -1)).reshape(input_mfcc.shape)
        input_mfcc_delta = self.scalers["input_features_mfcc_delta"].transform(input_mfcc_delta.reshape(1, -1)).reshape(input_mfcc_delta.shape)
        input_mfcc_delta2 = self.scalers["input_features_mfcc_delta2"].transform(input_mfcc_delta2.reshape(1, -1)).reshape(input_mfcc_delta2.shape)

        target_spectrogram = self.scalers["target_features_spectrogram"].transform(target_spectrogram.reshape(1, -1)).reshape(target_spectrogram.shape)

        # resample mfcc featues so theyre the same shape as the spectrogram and phase features
        # Resample MFCC features
        input_mfcc = self.resample_feature(input_mfcc, target_shape)
        input_mfcc_delta = self.resample_feature(input_mfcc_delta, target_shape)
        input_mfcc_delta2 = self.resample_feature(input_mfcc_delta2, target_shape)

        # Convert to tensors - input_phase, is missing,..... it's too confusing
        inputs = torch.tensor(np.stack([
            input_spectrogram, input_edges,
            input_mfcc, input_mfcc_delta, input_mfcc_delta2
        ], axis=0), dtype=torch.float32)  # Shape: (6, H, W)

        target = torch.tensor(target_spectrogram, dtype=torch.float32)  # Shape: (H, W)

        # Extract filename correctly
        filename = self.h5_file["filenames"][idx]
        if isinstance(filename, bytes):  # Check if it's a bytes object
            filename = filename.decode('utf-8')  # Convert to a string

        # Extract metadata
        metadata = {
            "filename": filename,
            "snr_db": self.h5_file["snr_db"][idx].item() # Convert to Python float
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