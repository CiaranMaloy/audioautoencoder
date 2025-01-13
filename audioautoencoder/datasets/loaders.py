import h5py
import torch
from torch.utils.data import Dataset

import h5py
import torch
from torch.utils.data import Dataset
import gcsfs

import h5py
import torch
from torch.utils.data import Dataset
import gcsfs

import h5py
import gcsfs
import torch
from torch.utils.data import Dataset

class HDF5DatasetGCS(Dataset):
    def __init__(self, bucket_name, file_path, output_time_length=84):
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.output_time_length = output_time_length
        self.fs = None  # Initialize GCSFileSystem as None
        self.input_key = "input_images"
        self.target_key = "target_images"
        self.num_samples = None  # Cache the dataset length for efficiency

    def _initialize_filesystem(self):
        """
        Initialize the GCS filesystem and fetch the dataset length if not already done.
        This ensures each worker process has its own GCSFileSystem instance.
        """
        if self.fs is None:
            self.fs = gcsfs.GCSFileSystem()
        
        if self.num_samples is None:
            with self.fs.open(f"{self.bucket_name}/{self.file_path}", 'rb') as f:
                with h5py.File(f, "r") as h5_file:
                    self.num_samples = h5_file[self.input_key].shape[0]

    def __len__(self):
        # Initialize the filesystem and fetch the dataset length
        self._initialize_filesystem()
        return self.num_samples

    def __getitem__(self, idx):
        # Ensure the filesystem is initialized in the worker process
        self._initialize_filesystem()
        with self.fs.open(f"{self.bucket_name}/{self.file_path}", 'rb') as f:
            with h5py.File(f, "r") as h5_file:
                input_image = h5_file[self.input_key][idx, :, :, :self.output_time_length]
                target_image = h5_file[self.target_key][idx, :, :, :self.output_time_length]
                return (
                    torch.tensor(input_image, dtype=torch.float32),
                    torch.tensor(target_image, dtype=torch.float32),
                )
     

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
