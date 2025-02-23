import torch
from tqdm import tqdm
from loss import *

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path='checkpoint.pt', mode='min'):
        """
        Early stopping to stop training when validation loss doesn't improve.

        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            save_path (str): Path to save the best model.
            mode (str): "min" or "max". Whether the monitored metric should be minimized or maximized.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'")

    def __call__(self, current_score, model, optimizer, epoch, total_epochs, loss):
        """
        Checks if early stopping should trigger.

        Args:
            current_score (float): The current value of the monitored metric.
            model (torch.nn.Module): The model to save if the current score is the best.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model, optimizer, epoch, total_epochs, loss)
        else:
            improvement = current_score - self.best_score if self.mode == 'max' else self.best_score - current_score
            if improvement > self.min_delta:
                self.best_score = current_score
                self.save_checkpoint(model, optimizer, epoch, total_epochs, loss)
                self.counter = 0
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model, optimizer, epoch, total_epochs, loss):
        """Saves the model when the validation score improves."""
        print(f"Validation score improved. Saving model to {self.save_path}.")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'entire_model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'total_epochs': total_epochs,
            'loss': loss
        }
        torch.save(checkpoint, self.save_path)

    def load_checkpoint(self, model):
        """Loads the best model."""
        model.load_state_dict(torch.load(self.save_path))

## Training, testing and calling examples
import matplotlib.pyplot as plt
import numpy as np

def print_loss_graph(losses):
  fig, ax = plt.subplots(figsize=(5, 2), dpi=100)

  # Set grey background
  fig.patch.set_facecolor("#333333")
  ax.set_facecolor("#333333")

  # Plot with a white line
  for i, loss in enumerate(losses):
    ax.plot(loss, color="white", lw=0.4, marker='x', linestyle=['solid', 'dashed', 'dotted'][i%3])

  # Make ticks visible and white
  ax.tick_params(axis='both', colors='white', length=2, width=0.2)

  # Set x-axis label, y-axis label, and plot title
  ax.set_xlabel('Epochs', color='white')
  ax.set_ylabel('Loss', color='white')
  ax.set_yscale('log')
  a = np.median(losses) + np.std(losses) * 2
  b = np.clip(np.median(losses) - np.std(losses) * 2, 0, 100)
  ax.set_ylim((b, a))
  ax.set_title('Loss over Epochs', color='white')

  # Set spines (border lines) to white
  for spine in ax.spines.values():
      spine.set_color('white')

  # Save the plot
  plt.show()

import os
import csv
import pandas as pd

class NoiseScheduler:
    def __init__(self, max_noise_std=0.2, min_noise_std=0.0, total_epochs=50, mode="linear"):
        """
        max_noise_std: Initial noise standard deviation
        min_noise_std: Final noise standard deviation (usually 0)
        total_epochs: Number of epochs over which noise reduces
        mode: "linear" or "exponential" decay
        """
        self.max_noise_std = max_noise_std
        self.min_noise_std = min_noise_std
        self.total_epochs = total_epochs
        self.mode = mode

    def get_noise_std(self, epoch):
        """Returns the noise level for a given epoch."""
        if self.mode == "linear":
            return max(
                self.min_noise_std,
                self.max_noise_std * (1 - epoch / self.total_epochs),
            )
        elif self.mode == "exponential":
            decay_rate = (self.min_noise_std / self.max_noise_std) ** (1 / self.total_epochs)
            return max(self.min_noise_std, self.max_noise_std * (decay_rate ** epoch))
        else:
            raise ValueError("Invalid mode. Choose 'linear' or 'exponential'.")

# Training loop
def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                early_stopping, 
                starting_epoch=0, 
                epochs=5, 
                verbose=False, 
                checkpoint_filename='checkpoint.pth', 
                scheduler_loss=False, 
                ref_min_value=0.4, 
                accumulation_steps=4, 
                max_noise=0.1,
                noise_epochs=10
                ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # reference loss
    reference_loss = MelWeightedMSELoss(device, min_value=ref_min_value)

    # Extract the directory and filename for saving logs
    checkpoint_dir = os.path.dirname(checkpoint_filename)
    # Create the directory if it doesn't exist
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    log_filename = os.path.join(checkpoint_dir, "training_log.csv")

    # If the log file doesn't exist, create it and write headers
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Learning Rate", "Train Loss", "Validation Loss", "Ref Loss", "KL Beta"])

    # step scheduler to correct training schedule
    if scheduler_loss:
        pass
    else:
        if starting_epoch > 0:
            for i in range(starting_epoch):
                print(f'Step {i+1}')
                scheduler.step()

    # introduce noise if specified
    noise_scheduler = NoiseScheduler(max_noise_std=max_noise, min_noise_std=0.0, total_epochs=noise_epochs, mode="linear")

    model.train()
    for epoch in range(starting_epoch, epochs):
        # Print the current learning rate
        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr}")

        # Train model
        model.train()
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        running_loss = 0.0
        recon_loss = 0.0
        ref_loss = 0.0

        # set loss beta
        beta = epoch/epochs
        print(f'New kl loss beta: {beta}')

        # get noise to add
        noise_std = noise_scheduler.get_noise_std(epoch)  # Get noise level for this epoch
        print('Noise Level: ', noise_std)

        i = 0
        if verbose:
          print('starting progress....')
        
        optimizer.zero_grad()
        for noisy_imgs, clean_imgs, _ in progress_bar:
            if verbose:
              print('in loop')
              progress_bar.set_description(f"Epoch {epoch + 1}, Batch {i}")
            noisy_imgs = noisy_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)
            if verbose:
              print('moving to device')
            
            if verbose:
              print('training model')

            # add noise to input
            noise = torch.randn_like(noisy_imgs) * noise_std
            noisy_imgs = noisy_imgs + noise

            # train model 
            outputs = model(noisy_imgs)

            if verbose:
                print(outputs.shape)
                print(clean_imgs.shape)

            true_loss = criterion(outputs, clean_imgs).item()  # Unscaled loss
            running_loss += true_loss

            loss = criterion(outputs, clean_imgs) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            #benchark_loss += criterion(noisy_imgs, clean_imgs).item()
            #recon_loss += r_loss.item()
            ref_loss += criterion(noisy_imgs[:, 0:2, :, :], clean_imgs[:, 0:2, :, :]).item() # this has been changed from 0:1 to 0:2
            progress_bar.set_postfix(loss=f"loss: {(running_loss) / (progress_bar.n + 1):.4f}, ref:{(ref_loss) / (progress_bar.n + 1):.4f}")
            #progress_bar.set_postfix(loss=f"{running_loss / (progress_bar.n + 1):.4f}, bl:{benchark_loss / (progress_bar.n + 1):.4f}")
            
            # i++
            i += 1

        # Validation step
        model.eval()
        progress_bar = tqdm(val_loader, desc="Validating", unit="batch")
        val_loss = 0.0
        recon_loss = 0.0
        with torch.no_grad():
            val_batch = 0
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                #recon_loss += r_loss.item()
                progress_bar.set_postfix(loss=f"joint loss: {(val_loss) / (progress_bar.n + 1):.4f}")
                #progress_bar.set_postfix(loss=f"{val_loss / (progress_bar.n + 1):.4f}")

        val_loss /= (len(progress_bar))
        
        if scheduler_loss:
            scheduler.step(val_loss)
        else:
            scheduler.step()


        print("-"*50)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        # Save training stats to CSV
        with open(log_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, current_lr, running_loss / len(train_loader), val_loss, ref_loss / len(train_loader), beta])

        # Check early stopping
        early_stopping(val_loss, model, optimizer, epoch, epochs, running_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            print("-"*50)
            break

        # saving model checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'entire_model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'total_epochs': epochs
        }
        torch.save(checkpoint, checkpoint_filename)
        print('Saved to Drive...')
        
        # plot loss graph
        df = pd.read_csv(log_filename)
        # Set the epoch column as the x-axis if it exists
        loss_type = ['Train Loss','Validation Loss','Ref Loss']
        losses = [df[l] for l in loss_type]
        print_loss_graph(losses)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
        print("-"*50)

import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
import torch.nn.init as init

# training class

class DenoisingTrainer:
    def __init__(self, model, noisy_train_loader, noisy_val_loader, 
                 SNRdB, output_path, patience=100, min_delta=0.0001, min_value=0.8, 
                 epochs=30, learning_rate=1e-3, load=True, warm_start=False, 
                 train=True, verbose=True, accumulation_steps=1, load_path=None, 
                 base_lr=1e-5, max_lr=1e-3, gamma=0.8, scheduler=None, optimizer=None, 
                 scheduler_loss=False, max_noise=0.05, noise_epochs=20
                 ):
        """Initialize the training environment with necessary parameters."""

        self.model = model
        self.train_loader = noisy_train_loader
        self.val_loader = noisy_val_loader
        self.SNRdB = SNRdB
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.load = load
        self.warm_start = warm_start
        self.train_flag = train
        self.verbose = verbose
        self.accumulation_steps = accumulation_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ref_min_value = min_value
        self.scheduler_loss = scheduler_loss
        self.max_noise = max_noise
        self.nosie_epochs = noise_epochs
        
        # Set up file paths
        self.output_path = output_path
        self.checkpoint_filename, self.earlystopping_filename = self.setup_directories(self.output_path)

        # load from file paths
        self.load_path = None
        if self.load == True:
            self.load_path = load_path
        else:
            model.apply(self.init_weights)  # Apply to entire model

        # Loss function (adjustable)
        self.criterion = nn.L1Loss()

        # Initialize Early Stopping
        self.early_stopping = EarlyStopping(
            patience=patience, min_delta=min_delta, save_path=self.earlystopping_filename, mode='min'
        )

        # Optimizer & Scheduler
        self.optimizer, self.scheduler = self.setup_optimizer_scheduler(base_lr, max_lr, gamma)
        
        # potential to define a scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        if optimizer is not None:
            self.optimizer = optimizer

        # Load checkpoint if required
        self.starting_epoch = 0
        if self.load:
            self.starting_epoch, self.epochs = self.load_checkpoint()

        # PyTorch memory configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def setup_directories(self, output_path):
        """Creates necessary directories and returns file paths."""
        checkpoint_filename = os.path.join(output_path, 'Autoencodermodel_checkpoint.pth')
        earlystopping_filename = os.path.join(output_path, 'Autoencodermodel_earlystopping.pth')

        os.makedirs(output_path, exist_ok=True)
        return checkpoint_filename, earlystopping_filename

    def setup_optimizer_scheduler(self, base_lr, max_lr, gamma, step_size_up=3):
        """Initializes the optimizer and learning rate scheduler."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='exp_range', gamma=gamma
        )
        return optimizer, scheduler

    def load_checkpoint(self):
        """Loads the latest checkpoint if available."""

        print(f'Loading model from: {self.load_path}')
        checkpoint = torch.load(self.load_path, map_location=self.device)

        print(f"Checkpoint keys: {checkpoint.keys()}")
        print(f"Loss: {checkpoint['loss']}, Epoch: {checkpoint['epoch']}, Total epochs: {checkpoint['total_epochs']}")

        starting_epoch = checkpoint['epoch'] if self.warm_start else 0
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return starting_epoch, checkpoint['total_epochs']
    
    def get_model(self):
        return self.model

    def train_or_evaluate(self):
        """Handles the training or evaluation process based on the flag."""
        if self.train_flag:
            train_model(
                self.model, self.train_loader, self.val_loader, self.criterion, 
                self.optimizer, self.scheduler, self.early_stopping, 
                starting_epoch=self.starting_epoch, epochs=self.epochs, verbose=self.verbose,
                checkpoint_filename=self.checkpoint_filename, ref_min_value=self.ref_min_value, 
                accumulation_steps=self.accumulation_steps, scheduler_loss=self.scheduler_loss, 
                max_noise=self.max_noise, noise_epochs=self.nosie_epochs
            )
        else:
            print('No training performed.')

if __name__ == '__main__':
    # --------------- Main Execution ---------------
    SNRdB = [-10, 10]  # Modify as needed
    output_path = ...

    # Initialize model and dataloaders here
    model = ...  # Define your model
    noisy_train_loader = ...  # Define your training dataloader
    noisy_val_loader = ...  # Define your validation dataloader

    trainer = DenoisingTrainer(
        model=model, noisy_train_loader=noisy_train_loader, noisy_val_loader=noisy_val_loader, 
        SNRdB=SNRdB, output_path=output_path, patience=100, min_delta=0.0001, min_value=0.8, epochs=30, learning_rate=1e-3, 
        load=True, warm_start=False, train=True, verbose=True, accumulation_steps=1
    )

    trainer.train_or_evaluate()


import torch
import torch.nn as nn

# class to load trained model
class DenoisingLoader:
    def __init__(self, model, checkpoint_path, device=None):
        """Loads a trained denoising model for inference."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Loads the model weights from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set model to evaluation mode
        print(f"Loaded model from {checkpoint_path}")

    def denoise(self, noisy_input):
        """Denoises an input tensor using the loaded model."""
        self.model.eval()
        with torch.no_grad():
            noisy_input = noisy_input.to(self.device)
            denoised_output = self.model(noisy_input)
        return denoised_output.cpu()