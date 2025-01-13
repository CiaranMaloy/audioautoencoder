import h5py
import torch
from torch.utils.data import Dataset

import h5py
import torch
from torch.utils.data import Dataset
import gcsfs

class HDF5DatasetGCS(Dataset):
    def __init__(self, bucket_name, file_path, output_time_length=86):
        # Initialize GCS filesystem
        self.fs = gcsfs.GCSFileSystem()
        
        # Store GCS file path and output length
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.output_time_length = output_time_length
        
        # Open the HDF5 file from GCS
        self.gcs_file = self.fs.open(f"{self.bucket_name}/{self.file_path}", 'rb')
        self.h5_file = h5py.File(self.gcs_file, "r")  # Open the file once
        
        # Access datasets
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
            # Access and trim the input and target images
            input_image = torch.tensor(self.input_dataset[idx, :, :, :self.output_time_length], dtype=torch.float32)
            target_image = torch.tensor(self.target_dataset[idx, :, :, :self.output_time_length], dtype=torch.float32)
            return input_image, target_image
        except Exception as e:
            # Log the error and index
            print(f"\nError loading sample {idx}: {e}")

            # Return placeholder tensors
            input_placeholder = torch.zeros(self.input_shape[1:], dtype=torch.float32)
            target_placeholder = torch.zeros(self.target_shape[1:], dtype=torch.float32)
            return input_placeholder, target_placeholder

    def __del__(self):
        # Close the HDF5 file and GCS file
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()
        if hasattr(self, "gcs_file") and self.gcs_file:
            self.gcs_file.close()

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, output_time_length=86):
        self.h5_file_path = h5_file_path
        self.output_time_length = output_time_length
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
