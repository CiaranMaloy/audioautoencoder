import torch
from torch.utils.data import DataLoader, random_split
from datasets.loaders import *

class NoisyDatasetLoader:
    def __init__(self, dataset_path, output_time_length=175, channels=1, snr_db=None, subset=False, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = 0
        self.subset = subset
        self.snr_db = snr_db
        self.dataset_path = dataset_path
        self.output_time_length = output_time_length
        self.channels = channels
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        torch.manual_seed(42)
        split_rng = torch.Generator().manual_seed(42)
        dataset = HDF5Dataset_metadata(self.dataset_path, output_time_length=self.output_time_length, channels=self.channels)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=split_rng)
        
        if self.subset:
            self._apply_subset(split_rng)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
    
    def _apply_subset(self, split_rng):
        subsample = 0.35
        self.train_dataset, _ = random_split(self.train_dataset, [int(subsample * len(self.train_dataset)), len(self.train_dataset) - int(subsample * len(self.train_dataset))], generator=split_rng)
        self.val_dataset, _ = random_split(self.val_dataset, [int(subsample * len(self.val_dataset)), len(self.val_dataset) - int(subsample * len(self.val_dataset))], generator=split_rng)

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
