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


import torch
import numpy as np

def custom_collate_fn(batch):
    # Unzip the batch into inputs, targets, and metadata
    inputs, targets, metadata = zip(*batch)
    
    # Return the batch as a tuple of tensors and metadata
    return inputs, targets, metadata