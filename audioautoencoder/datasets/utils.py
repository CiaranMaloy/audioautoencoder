import torch
from torch.utils.data import DataLoader, random_split
from datasets.loaders import *

class NoisyDatasetLoader:
    def __init__(self, dataset_path, output_time_length=175, channels=1, snr_db=None, subset=False, batch_size=32, metadata=False, features=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = 0
        self.subset = subset
        self.snr_db = snr_db
        self.dataset_path = dataset_path
        self.output_time_length = output_time_length
        self.channels = channels
        self.metadata = metadata
        self.features = features
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        torch.manual_seed(42)
        split_rng = torch.Generator().manual_seed(42)

        if self.metadata:
            dataset = HDF5Dataset_metadata(self.dataset_path, output_time_length=self.output_time_length, channels=self.channels)
        elif self.features:
            dataset = HDF5Dataset_features(self.dataset_path, output_time_length=self.output_time_length, channels=self.channels)
        else:
            dataset = HDF5Dataset(self.dataset_path, output_time_length=self.output_time_length, channels=self.channels)
        
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=split_rng)
        
        if self.subset:
            self._apply_subset(split_rng)
        
        if self.metadata:
            # include custom colate function for metadata
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        elif self.features:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
    
    def _apply_subset(self, split_rng):
        subsample = 0.35
        self.train_dataset, _ = random_split(self.train_dataset, [int(subsample * len(self.train_dataset)), len(self.train_dataset) - int(subsample * len(self.train_dataset))], generator=split_rng)
        self.val_dataset, _ = random_split(self.val_dataset, [int(subsample * len(self.val_dataset)), len(self.val_dataset) - int(subsample * len(self.val_dataset))], generator=split_rng)

class ScalerDatasetLoader:
    def __init__(self, dataset_path, scalers, output_time_length=175, channels=1, snr_db=None, subset=False, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = 0
        self.subset = subset
        self.snr_db = snr_db
        self.dataset_path = dataset_path
        self.scalers = scalers
        self.output_time_length = output_time_length
        self.channels = channels
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        torch.manual_seed(42)
        split_rng = torch.Generator().manual_seed(42)

        dataset = HDF5Dataset_features(self.dataset_path, self.scalers, output_time_length=self.output_time_length, channels=self.channels)
    
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=split_rng)
        
        if self.subset:
            self._apply_subset(split_rng)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
    
    def _apply_subset(self, split_rng):
        subsample = 0.35
        self.train_dataset, _ = random_split(self.train_dataset, [int(subsample * len(self.train_dataset)), len(self.train_dataset) - int(subsample * len(self.train_dataset))], generator=split_rng)
        self.val_dataset, _ = random_split(self.val_dataset, [int(subsample * len(self.val_dataset)), len(self.val_dataset) - int(subsample * len(self.val_dataset))], generator=split_rng)

class ChannelDatasetLoader:
    def __init__(self, dataset_path, scalers, output_time_length=175, channels=1, snr_db=None, subset=False, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = 0
        self.subset = subset
        self.snr_db = snr_db
        self.dataset_path = dataset_path
        self.scalers = scalers
        self.output_time_length = output_time_length
        self.channels = channels
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        torch.manual_seed(42)
        split_rng = torch.Generator().manual_seed(42)

        dataset = HDF5Dataset_bandchannels(self.dataset_path, self.scalers, output_time_length=self.output_time_length, channels=self.channels)
    
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=split_rng)
        
        if self.subset:
            self._apply_subset(split_rng)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
    
    def _apply_subset(self, split_rng):
        subsample = 0.35
        self.train_dataset, _ = random_split(self.train_dataset, [int(subsample * len(self.train_dataset)), len(self.train_dataset) - int(subsample * len(self.train_dataset))], generator=split_rng)
        self.val_dataset, _ = random_split(self.val_dataset, [int(subsample * len(self.val_dataset)), len(self.val_dataset) - int(subsample * len(self.val_dataset))], generator=split_rng)


class DiffusionDatasetLoader:
    def __init__(self, dataset_path, scalers, output_time_length=175, channels=1, snr_db=None, subset=False, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = 0
        self.subset = subset
        self.snr_db = snr_db
        self.dataset_path = dataset_path
        self.scalers = scalers
        self.output_time_length = output_time_length
        self.channels = channels
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        torch.manual_seed(42)
        split_rng = torch.Generator().manual_seed(42)

        dataset = HDF5Dataset_bandchannels_diffusion(self.dataset_path, self.scalers, output_time_length=self.output_time_length, channels=self.channels)
    
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=split_rng)
        
        if self.subset:
            self._apply_subset(split_rng)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
        
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
    
    def _apply_subset(self, split_rng):
        subsample = 0.35
        self.train_dataset, _ = random_split(self.train_dataset, [int(subsample * len(self.train_dataset)), len(self.train_dataset) - int(subsample * len(self.train_dataset))], generator=split_rng)
        self.val_dataset, _ = random_split(self.val_dataset, [int(subsample * len(self.val_dataset)), len(self.val_dataset) - int(subsample * len(self.val_dataset))], generator=split_rng)

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def train_scalers(dataset_path, sample_size=1000):
    """Trains scalers for each feature in the HDF5 dataset."""
    scalers = {}
    
    with h5py.File(dataset_path, "r") as source_file:
        
        def get_scaler(data, scaler, sample_size=1000):
          """Fit scaler on a random subset of data to speed up training."""
          num_samples = data.shape[0]
          sample_size = min(sample_size, num_samples)  # Ensure we don't exceed available samples

          # Randomly select indices
          indices = np.sort(np.random.choice(num_samples, size=sample_size, replace=False))
          sampled_data = data[indices]  # Efficiently select rows

          # Fit scaler
          return scaler.fit(sampled_data.reshape(sampled_data.shape[0], -1))  # Flatten for scaler

        # Train scalers
        # input
        print('Training Phase...')
        scalers["input_features_phase"] = get_scaler(source_file["input_features_phase"], MinMaxScaler(), sample_size=sample_size)
        print('Training Spectrogram...')
        scalers["input_features_spectrogram"] = get_scaler(source_file["input_features_spectrogram"], StandardScaler(), sample_size=sample_size)
        print('Training Edges...')
        scalers["input_features_edges"] = get_scaler(source_file["input_features_edges"], StandardScaler(), sample_size=sample_size)
        print('Training Cepstrum...')
        scalers["input_features_cepstrum"] = get_scaler(source_file["input_features_cepstrum"], StandardScaler(), sample_size=sample_size)
        print('Training Cepstrum Edges...')
        scalers["input_features_cepstrum_edges"] = get_scaler(source_file["input_features_cepstrum_edges"], StandardScaler(), sample_size=sample_size)
        # target
        print('Training Target Spectrogram...')
        scalers["target_features_spectrogram"] = get_scaler(source_file["target_features_spectrogram"], StandardScaler(), sample_size=sample_size)
        
    return scalers

def get_scaler(data, scaler, sample_size=1000):
          """Fit scaler on a random subset of data to speed up training."""
          num_samples = data.shape[0]
          sample_size = min(sample_size, num_samples)  # Ensure we don't exceed available samples

          # Randomly select indices
          indices = np.sort(np.random.choice(num_samples, size=sample_size, replace=False))
          sampled_data = data[indices]  # Efficiently select rows

          # Fit scaler
          return scaler.fit(sampled_data.reshape(sampled_data.shape[0], -1))  # Flatten for scaler

def get_scaler_partial(data, scaler, sample_size=1000):
          """Fit scaler on a random subset of data to speed up training."""
          num_samples = data.shape[0]
          sample_size = min(sample_size, num_samples)  # Ensure we don't exceed available samples

          # Randomly select indices
          indices = np.sort(np.random.choice(num_samples, size=sample_size, replace=False))
          sampled_data = data[indices]  # Efficiently select rows

          # Fit scaler
          return scaler.partial_fit(sampled_data.reshape(sampled_data.shape[0], -1))  # Flatten for scaler


def train_scalers_separation(dataset_path, sample_size=1000):
    print('Training scalers for separation dataset')
    """Trains scalers for each feature in the HDF5 dataset."""
    scalers = {}
    
    with h5py.File(dataset_path, "r") as source_file:


        # train spectrogram scaler
        print('Training Spectrogram...')
        spec_scaler = StandardScaler()
        print('Input features')
        spec_scaler = get_scaler_partial(source_file["input_features_spectrogram"], spec_scaler, sample_size=sample_size)
        print('Target features')
        spec_scaler = get_scaler_partial(source_file["target_features_spectrogram"], spec_scaler, sample_size=sample_size)

        # Train scalers
        # input
        #print('Training Phase...')
        #scalers["input_features_phase"] = get_scaler(source_file["input_features_phase"], MinMaxScaler(), sample_size=sample_size)

        scalers["input_features_spectrogram"] = spec_scaler
        print('Training Edges...')
        scalers["input_features_edges"] = get_scaler(source_file["input_features_edges"], StandardScaler(), sample_size=sample_size)
        print('Training Cepstrum...')
        scalers["input_features_cepstrum"] = get_scaler(source_file["input_features_cepstrum"], StandardScaler(), sample_size=sample_size)
        print('Training Cepstrum Edges...')
        scalers["input_features_cepstrum_edges"] = get_scaler(source_file["input_features_cepstrum_edges"], StandardScaler(), sample_size=sample_size)
        # target
        scalers["target_features_spectrogram"] = spec_scaler
        
    return scalers

def train_scalers_diffusion(dataset_path, sample_size=1000):
    print('Training scalers for separation dataset')
    """Trains scalers for each feature in the HDF5 dataset."""
    scalers = {}
    
    with h5py.File(dataset_path, "r") as source_file:


        # train spectrogram scaler
        print('Training Spectrogram...')
        spec_scaler = StandardScaler()
        print('Input features')
        #spec_scaler = get_scaler_partial(source_file["input_features_spectrogram"], spec_scaler, sample_size=sample_size)
        print('Target features')
        spec_scaler = get_scaler_partial(source_file["target_features_spectrogram"], spec_scaler, sample_size=sample_size)

        # Train scalers
        # input
        #print('Training Phase...')
        #scalers["input_features_phase"] = get_scaler(source_file["input_features_phase"], MinMaxScaler(), sample_size=sample_size)

        scalers["input_features_spectrogram"] = spec_scaler
        print('Training Edges...')
        #scalers["input_features_edges"] = get_scaler(source_file["input_features_edges"], StandardScaler(), sample_size=sample_size)
        print('Training Cepstrum...')
        #scalers["input_features_cepstrum"] = get_scaler(source_file["input_features_cepstrum"], StandardScaler(), sample_size=sample_size)
        print('Training Cepstrum Edges...')
        #scalers["input_features_cepstrum_edges"] = get_scaler(source_file["input_features_cepstrum_edges"], StandardScaler(), sample_size=sample_size)
        # target
        scalers["target_features_spectrogram"] = spec_scaler
        
    return scalers

import os
import joblib  # or use pickle if you prefer

def save_scalers(scalers, save_path):
    """Save scalers to a file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    joblib.dump(scalers, save_path)

def load_scalers(save_path):
    """Load scalers from a file."""
    return joblib.load(save_path)

def resample_feature(feature, target_shape):
    """Resamples a 2D numpy feature array to match target shape using torch.nn.functional.interpolate."""
    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    target_size = (target_shape[0], target_shape[1])  # (new_H, new_W)
    
    resized_feature = F.interpolate(feature_tensor, size=target_size, mode="bilinear", align_corners=False)
    return resized_feature.squeeze(0).squeeze(0).numpy()  # Remove batch/channel dim and return as numpy

if __name__ == '__main__':
    SNRdB = [-10, 40]
    IMPORT_TRAIN_NOISY = True
    if IMPORT_TRAIN_NOISY:
            loader = NoisyDatasetLoader(
                dataset_path=f"/content/SNRdB_{SNRdB[0]}-{SNRdB[1]}/train/combined_000.h5",
                output_time_length=175,
                channels=1,
                snr_db=SNRdB,
                subset=False
            )
