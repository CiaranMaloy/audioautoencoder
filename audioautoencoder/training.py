import torch
from tqdm import tqdm

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

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, starting_epoch=0, epochs=5, max_val_batches=30, verbose=False, checkpoint_filename='checkpoint.pth', scheduler_loss=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model.train()
    for epoch in range(starting_epoch, epochs):
        # Print the current learning rate
        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr}")

        # Train model
        model.train()
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        running_loss = 0.0
        i = 0
        if verbose:
          print('starting progress....')
        for noisy_imgs, clean_imgs in progress_bar:
            if verbose:
              print('in loop')
              progress_bar.set_description(f"Epoch {epoch + 1}, Batch {i}")
            noisy_imgs = noisy_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)
            if verbose:
              print('moving to device')
            
            optimizer.zero_grad()
            if verbose:
              print('training model')
            outputs, mask = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i += 1
            progress_bar.set_postfix(loss=f"{running_loss / (progress_bar.n + 1):.4f}")

        # Validation step
        model.eval()
        progress_bar = tqdm(val_loader, desc="Validating", unit="batch")
        val_loss = 0.0
        with torch.no_grad():
            val_batch = 0
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, mask = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                progress_bar.set_postfix(loss=f"{val_loss / (progress_bar.n + 1):.4f}")
                if val_batch <= max_val_batches:
                  val_batch += 1
                else:
                  break
        val_loss /= val_batch
        
        if scheduler_loss:
            scheduler.step(val_loss)
        else:
            scheduler.step()


        print("-"*50)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

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
            outputs, mask = model(noisy_imgs)
            break  # Display only the first batch

    return noisy_imgs.cpu(), outputs.cpu(), clean_imgs.cpu()
