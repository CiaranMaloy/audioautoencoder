import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from processing import *

def process_and_reconstruct_audio(
    denoised_imgs, noisy_imgs, clean_imgs, image_to_waveform, sr=44100, i=0, cmap='twilight'
):
    """
    Processes spectrogram images and reconstructs audio for visualization and comparison.

    Parameters:
    - denoised_imgs: Array of denoised spectrogram images [shape: (N, 2, H, W)].
    - noisy_imgs: Array of noisy spectrogram images [shape: (N, 2, H, W)].
    - clean_imgs: Array of clean spectrogram images [shape: (N, 2, H, W)].
    - image_to_waveform: Function to convert spectrogram images to waveform audio.
    - sr: Sample rate for spectrograms (default: 44100).
    - i: Index of the image to process (default: 0).
    - cmap: Colormap for visualization (default: 'twilight').

    Returns:
    - clean_audio: Reconstructed clean audio waveform.
    - noisy_audio: Reconstructed noisy audio waveform.
    - new_audio: Reconstructed denoised audio waveform.
    """
    # Extract images
    image = denoised_imgs[i]
    image_noisy = noisy_imgs[i]
    image_clean = clean_imgs[i]

    print("Max value in denoised image:", np.max(image))
    print("Min value in denoised image:", np.min(image))

    # Plotting spectrograms
    def plot_spectrograms(images, title_prefix):
        plt.figure(figsize=(12, 8))
        for idx, img in enumerate(images):
            plt.subplot(3, 1, idx + 1)
            librosa.display.specshow(img, sr=sr, y_axis='log', x_axis='time', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'{title_prefix} - {["Magnitude", "Normalized", "Phase"][idx]} Spectrogram')
        plt.tight_layout()
        plt.show()

    plot_spectrograms(image, "Denoised")
    plot_spectrograms(image_noisy, "Noisy")
    plot_spectrograms(image_clean, "Clean")

    # Plot mean values along the frequency axis
    def plot_mean_along_frequency_axis(image_data, label):
        plt.plot(np.mean(image_data[0], axis=1), label=f'{label} - Magnitude')
        plt.plot(np.mean(image_data[1], axis=1), label=f'{label} - Normalized')
        plt.legend()
        plt.show()

    plot_mean_along_frequency_axis(image_noisy, "Noisy")
    plot_mean_along_frequency_axis(image_clean, "Clean")

    # Reconstruct audio from images
    if image_to_waveform:
        new_audio = image_to_waveform(image, sr)
        new_audio = new_audio / np.max(np.abs(new_audio))
        noisy_audio = image_to_waveform(image_noisy, sr)
        noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
        clean_audio = image_to_waveform(image_clean, sr)
        clean_audio = clean_audio / np.max(np.abs(clean_audio))

        print("Audio processed!")

        # Plot waveforms
        n = 0
        c = len(clean_audio)
        def plot_waveform(audio, title):
            plt.plot(audio)
            plt.xlim((n, n + c))
            plt.title(title)
            plt.show()

        plot_waveform(clean_audio, "Clean Audio")
        plot_waveform(new_audio, "Denoised Audio")
        plot_waveform(noisy_audio, "Noisy Audio")

        return clean_audio, noisy_audio, new_audio
    

def plot_specgrams_separate(
    noisy_imgs, clean_imgs, sr=44100, i=0, cmap='twilight'
):
    """
    Processes spectrogram images and reconstructs audio for visualization and comparison.

    Parameters:
    - denoised_imgs: Array of denoised spectrogram images [shape: (N, 2, H, W)].
    - noisy_imgs: Array of noisy spectrogram images [shape: (N, 2, H, W)].
    - clean_imgs: Array of clean spectrogram images [shape: (N, 2, H, W)].
    - image_to_waveform: Function to convert spectrogram images to waveform audio.
    - sr: Sample rate for spectrograms (default: 44100).
    - i: Index of the image to process (default: 0).
    - cmap: Colormap for visualization (default: 'twilight').

    Returns:
    - clean_audio: Reconstructed clean audio waveform.
    - noisy_audio: Reconstructed noisy audio waveform.
    - new_audio: Reconstructed denoised audio waveform.
    """
    # Extract images
    image_noisy = noisy_imgs[i]
    image_clean = clean_imgs[i]

    print("Noisy Image:")
    print(f"Max: {np.max(image_noisy[0])}, Min: {np.min(image_noisy[0])}")
    print(f"Mean: {np.mean(image_noisy[0])}, Std: {np.std(image_noisy[0])}")

    print("\nClean Image:")
    print(f"Max: {np.max(image_clean[0])}, Min: {np.min(image_clean[0])}")
    print(f"Mean: {np.mean(image_clean[0])}, Std: {np.std(image_clean[0])}")

    print("\nNoise Image:")
    print(f"Max: {np.max(image_clean[1])}, Min: {np.min(image_clean[1])}")
    print(f"Mean: {np.mean(image_clean[1])}, Std: {np.std(image_clean[1])}")

    # Plotting spectrograms
    def plot_spectrograms(images, title_prefix):
        plt.figure(figsize=(12, 8))
        for idx, img in enumerate(images):
            plt.subplot(3, 1, idx + 1)
            librosa.display.specshow(img, sr=sr, y_axis='log', x_axis='time', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'{title_prefix} - {["1", "2"][idx]} Spectrogram')
        plt.tight_layout()
        plt.show()

    #plot_spectrograms(image, "Denoised")
    plot_spectrograms(image_noisy, "Noisy")
    plot_spectrograms(image_clean, "Clean")

    # Plot mean values along the frequency axis
    def plot_mean_along_frequency_axis(image_data, label):
        plt.plot(np.mean(image_data[0], axis=1), label=f'{label} - 1')
        plt.plot(np.mean(image_data[1], axis=1), label=f'{label} - 2')
        plt.legend()
        plt.show()

    plot_mean_along_frequency_axis(image_noisy, "Noisy")
    plot_mean_along_frequency_axis(image_clean, "Clean")

    mix_waveform = magphase_to_waveform(image_noisy[0], image_noisy[1], audio_length=44100*2)
    clean_waveform = magphase_to_waveform(image_clean[0], image_noisy[1], audio_length=44100*2)
    noise_waveform = magphase_to_waveform(image_clean[1], image_noisy[1], audio_length=44100*2)

    return mix_waveform, clean_waveform, noise_waveform


import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

def plot_image_means(image, title="Image Means"):
    """Plots the mean values of the input image along the first axis."""
    plt.figure()
    for channel in range(image.shape[0]):
        plt.plot(np.mean(image[channel], axis=1), label=f'Channel {channel}')
    plt.legend()
    plt.title(title)
    plt.show()

def process_and_visualize_audio(noise_level=0.00001, sr=44100, save_path=None):
    """Processes audio, converts it to image format, visualizes it, and saves output waveforms."""
    # Generate input audio
    audio, sr = generate_input_audio()
    
    # Convert to image representation
    input_image, target_image = process_audio_to_image(audio, sr, plot=True, noise_level=noise_level)
    
    # Plot input and target image mean values
    plot_image_means(input_image, "Input Image Means")
    plot_image_means(target_image, "Target Image Means")
    
    # Convert image back to waveform
    input_waveform = image_to_waveform(input_image, sr)
    target_waveform = image_to_waveform(target_image, sr)
    
    # Print statistics
    for i, name in enumerate(["input_image", "target_image"]):
        for channel in range(input_image.shape[0]):
            print(f"{name}[{channel}] max: {np.max(input_image[channel])}, min: {np.min(input_image[channel])}")
    
    print("Input image shape:", np.shape(input_image))
    
    # Plot waveforms
    def plot_waveform(waveform, title):
        plt.figure()
        plt.plot(waveform)
        plt.xlim((4000, 8000))
        plt.title(title)
        plt.show()
    
    plot_waveform(input_waveform, "Input Waveform")
    plot_waveform(target_waveform, "Target Waveform")
    
    # Ensure save directory exists
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save processed waveforms
        sf.write(f"{save_path}/test_input.wav", input_waveform, sr)
        sf.write(f"{save_path}/test_target.wav", target_waveform, sr)

import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from torch.optim import SGD

def plot_learning_rate(base_lr, max_lr, gamma, n, step_size_up=3):
    
  # Dummy optimizer
  model_params = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
  optimizer = SGD(model_params, lr=1e-3)

  # Define CyclicLR scheduler
  scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='exp_range', gamma=gamma)

  # Simulate 30 epochs
  lrs = []
  for epoch in range(n):
      optimizer.step()  # Dummy step
      lrs.append(optimizer.param_groups[0]['lr'])
      scheduler.step()  # Update scheduler

  # Plot the learning rates
  plt.figure(figsize=(8, 5))
  plt.plot(range(1, n+1), lrs, marker='o', linestyle='-', label="Learning Rate")
  plt.title("CyclicLR Scheduler over 30 Epochs")
  plt.xlabel("Epoch")
  plt.ylabel("Learning Rate")
  plt.yscale('log')
  plt.grid(True)
  plt.legend()
  plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
def plot_training_log(csv_file_path):
  df = pd.read_csv(csv_file_path)

  # Check the first few rows to confirm structure
  print(df.head())

  # Set the epoch column as the x-axis if it exists
  x = range(len(df))  # Fallback if epoch column is missing

  # Plot all numerical columns
  plt.figure(figsize=(10, 6))
  for col in df.select_dtypes(include=['number']).columns:
      if col != 'Epoch' and col != 'KL Beta':  # Skip epoch column in y-axis plots
          plt.plot(x, df[col], label=col)

  # Formatting
  plt.xlabel("Epoch")
  plt.ylabel("Loss / Metrics")
  plt.title("Training and Validation Metrics Over Time")
  plt.legend()
  plt.yscale('log')
  #plt.ylim((0, 0.03))
  plt.grid(True)

  # Show plot
  plt.show()

def plot_spectrograms_at_timesteps(model, train_loader, diffusion_scheduler, timesteps=[999, 300, 200, 50, 1]):
    """
    Plot spectrograms at different diffusion timesteps.
    
    Args:
        model: Your diffusion model
        train_loader: DataLoader for training data
        diffusion_scheduler: DDPM_Scheduler instance
        timesteps: List of timesteps to visualize
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Get a batch from the dataloader
    for _, clean_imgs, _ in train_loader:
        clean_imgs = clean_imgs.to(device)
        break  # Just get one batch
    
    # Choose a random example from the batch
    example_idx = np.random.randint(0, clean_imgs.shape[0])
    print(np.shape(clean_imgs))
    example = clean_imgs[3:4]  # Keep batch dimension
    print(np.shape(example))
    
    plt.figure(figsize=(15, 10))
    
    # Plot original clean spectrogram
    plt.subplot(len(timesteps) + 1, 1, 1)
    plt.title('Original Clean Spectrogram')
    plt.imshow(example[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    
    # Plot for each timestep
    for i, t in enumerate(timesteps):
        # Create a tensor with the same timestep for the batch
        t_tensor = torch.tensor([t], device=device)
        
        # Add noise according to the scheduler
        noise = torch.randn_like(example)
        a = diffusion_scheduler.alpha[t]
        a_tensor = torch.tensor(a, device=device).view(1, 1, 1, 1)
        
        # Apply forward diffusion process
        noisy_example = (torch.sqrt(a_tensor) * example) + (torch.sqrt(1 - a_tensor) * noise)
        
        # Plot the noisy spectrogram
        plt.subplot(len(timesteps) + 1, 1, i + 2)
        plt.title(f'Timestep t={t}, α={a:.4f}')
        plt.imshow(noisy_example[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# plot while training: 
from einops import rearrange 

def inference(model, ema, input_tensor, starting_timestep, scheduler, times):

    # Move input tensor to GPU
    z = input_tensor.cuda()
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        for t in reversed(range(1, starting_timestep)):
            t = [t]
            # Ensure scheduler.beta and alpha are on GPU (move them once outside the loop if possible)
            beta_t = scheduler.beta[t].cuda()
            alpha_t = scheduler.alpha[t].cuda()
            
            temp = (beta_t / (torch.sqrt(1 - alpha_t) * torch.sqrt(1 - beta_t)))
            z = (1 / torch.sqrt(1 - beta_t)) * z - (temp * model(z, [0]))  # z is already on GPU, model output is on GPU
            if t[0] in times:
                images.append(z)
            # Ensure noise tensor e is on GPU
            e = torch.randn(1, 4, 1025, 175).cuda() #[1, 4, 1025, 175]
            z = z + (e * torch.sqrt(beta_t))
        
        # Final step
        beta_0 = scheduler.beta[0].cuda()
        alpha_0 = scheduler.alpha[0].cuda()
        temp = beta_0 / (torch.sqrt(1 - alpha_0) * torch.sqrt(1 - beta_0))
        x = (1 / torch.sqrt(1 - beta_0)) * z - (temp * model(z, [0]))  # z and model output are on GPU
        x = x.cpu()  # Move to CPU only at the end

        #images.append(x)
        x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
        x = x.numpy()
        
    return images

def plot_spectrograms_at_timesteps_training_validation(model, val_loader, diffusion_scheduler, ema):
    """
    Plot spectrograms at different diffusion timesteps.
    
    Args:
        model: Your diffusion model
        train_loader: DataLoader for training data
        diffusion_scheduler: DDPM_Scheduler instance
        timesteps: List of timesteps to visualize
    """
    timesteps=[999, 400, 300, 200, 50] # hardcoded
    reconstruction_view = [[500, 200, 50, 1], [300, 150, 25, 1], [200, 75, 50, 1], [100, 50, 25, 1], [30, 15, 7, 1]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Get a batch from the dataloader
    for _, clean_imgs, _ in val_loader:
        clean_imgs = clean_imgs.to(device)
        break  # Just get one batch
    
    # Choose a random example from the batch
    example_idx = np.random.randint(0, clean_imgs.shape[0])
    example = clean_imgs[example_idx:example_idx+1] 
    
    plt.figure(figsize=(15, 15))
    
    # Plot for each timestep
    for i, t in enumerate(timesteps):
        # Create a tensor with the same timestep for the batch
        t_tensor = torch.tensor([t], device=device)
        
        # Add noise according to the scheduler
        noise = torch.randn_like(example)
        a = diffusion_scheduler.alpha[t]
        a_tensor = torch.tensor(a, device=device).view(1, 1, 1, 1)
        
        # Apply forward diffusion process
        noisy_example = (torch.sqrt(a_tensor) * example) + (torch.sqrt(1 - a_tensor) * noise)
        
        # Plot the noisy spectrogram
        plt.subplot(len(timesteps), len(timesteps), i * len(timesteps) + 1)
        plt.title(f'Timestep t={t}, α={a:.4f}')
        plt.imshow(noisy_example[0, 3].cpu().detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()

        # 1. process noisy_example
        images = inference(model, ema, noisy_example[0:1], t, diffusion_scheduler, reconstruction_view[i])


        # 2. plot views
        for k, image in enumerate(images):
            plt.subplot(len(timesteps), len(timesteps), i*len(timesteps) + k + 2)
            plt.title(f'Timestep t={reconstruction_view[i][k]}')
            plt.imshow(image[0, 3].cpu().detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == '__main__':
    CHECK = False
    if CHECK:
        process_and_visualize_audio(
            noise_level=0.00001,
            sr=44100,
            save_path='/content/drive/MyDrive/Projects/ML_Projects/De-noising-autoencoder/Models/UNetDenoisingAutoencoder/Examples/'
        )
