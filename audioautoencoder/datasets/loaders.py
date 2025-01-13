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

class HDF5DatasetGCS(Dataset):
    def __init__(self, bucket_name, file_path, output_time_length=86):
        """
        Initialize the dataset by specifying the GCS bucket and file path.
        """
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.output_time_length = output_time_length
        self.fs = gcsfs.GCSFileSystem()  # Initialize GCS filesystem

        # Preload dataset shapes
        with self.fs.open(f"gs://{self.bucket_name}/{self.file_path}", 'rb') as f:
            with h5py.File(f, "r") as h5_file:
                self.input_shape = h5_file["input_images"].shape
                self.target_shape = h5_file["target_images"].shape

        # Verify dataset alignment
        assert self.input_shape[0] == self.target_shape[0], "Input and target datasets are misaligned."

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.input_shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a single data sample by index.
        """
        try:
            with self.fs.open(f"gs://{self.bucket_name}/{self.file_path}", 'rb') as f:
                with h5py.File(f, "r") as h5_file:
                    input_image = torch.tensor(
                        h5_file["input_images"][idx, :, :, :self.output_time_length],
                        dtype=torch.float32
                    )
                    target_image = torch.tensor(
                        h5_file["target_images"][idx, :, :, :self.output_time_length],
                        dtype=torch.float32
                    )
            return input_image, target_image
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return placeholder tensors in case of an error
            input_placeholder = torch.zeros((self.input_shape[1], self.input_shape[2], self.output_time_length), dtype=torch.float32)
            target_placeholder = torch.zeros((self.target_shape[1], self.target_shape[2], self.output_time_length), dtype=torch.float32)
            return input_placeholder, target_placeholder

        

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
