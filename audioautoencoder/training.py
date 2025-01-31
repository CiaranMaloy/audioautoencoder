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

import os
import csv
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
                accumulation_steps=4
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

        i = 0
        if verbose:
          print('starting progress....')
        
        optimizer.zero_grad()
        for noisy_imgs, clean_imgs in progress_bar:
            if verbose:
              print('in loop')
              progress_bar.set_description(f"Epoch {epoch + 1}, Batch {i}")
            noisy_imgs = noisy_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)
            if verbose:
              print('moving to device')
            
            if verbose:
              print('training model')
            outputs = model(noisy_imgs)

            if verbose:
                print(outputs.shape)
                print(clean_imgs.shape)
            loss = criterion(outputs, clean_imgs) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            #benchark_loss += criterion(noisy_imgs, clean_imgs).item()

            running_loss += loss.item() * accumulation_steps
            #recon_loss += r_loss.item()
            ref_loss += criterion(noisy_imgs[:, 0:1, :, :], clean_imgs[:, 0:1, :, :]).item()
            progress_bar.set_postfix(loss=f"loss: {running_loss / (progress_bar.n + 1):.4f}, ref:{ref_loss / (progress_bar.n + 1):.4f}")
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
                progress_bar.set_postfix(loss=f"joint loss: {val_loss / (progress_bar.n + 1):.4f}")
                #progress_bar.set_postfix(loss=f"{val_loss / (progress_bar.n + 1):.4f}")

        val_loss /= (len(progress_bar) + 1)
        
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
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'total_epochs': epochs
        }
        torch.save(checkpoint, checkpoint_filename)
        print('Saved to Drive...')
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
        print("-"*50)

# Testing loop
def test_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for inputs, targets in progress_bar:

          inputs, targets = inputs.to(device), targets.to(device)
          outputs, mask = model(inputs)
          loss = criterion(outputs, targets)
          progress_bar.set_postfix(loss=f"{loss.item():.4f}")
          test_loss += loss.item()

        test_loss /= len(test_loader)

    return test_loss

# Testing loop
def test_examples(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            #input_features = input_features.to(device)
            clean_imgs = clean_imgs.to(device)
            outputs = model(noisy_imgs)
            break  # Display only the first batch

    return noisy_imgs.cpu(), outputs.cpu(), clean_imgs.cpu()
