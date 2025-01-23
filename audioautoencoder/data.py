# Core libraries
import os
import random
from pathlib import Path
import psutil

# Numerical and scientific computing
import numpy as np

# Audio processing
import librosa
import torchaudio
import soundfile as sf

# File handling and data storage
import h5py

# Progress bar
from tqdm import tqdm

# Parallel processing
from concurrent.futures import ProcessPoolExecutor

# plotting
import matplotlib.pyplot as plt

# import from processing
from processing import *


# try batch processing first before I try just waiting for 13 hours
def is_valid_audio(file_path):
    """
    Check if the audio file can be loaded successfully.
    Returns True if valid, False if not.
    """
    try:
        # Attempt to load the audio file to check validity
        waveform, sr = torchaudio.load(file_path)
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

def gather_valid_audio_files(data_dir):
    """
    Gather all valid audio files in the directory, using tqdm to show progress.

    Args:
        data_dir (str): Path to the directory containing audio files.

    Returns:
        List of valid audio files.
    """
    # Get the list of .wav files in the directory
    print('Gathering wav files....')
    wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

    total_files = len(wav_files)
    if total_files == 0:
        print("No .wav files found in the specified directory.")
        return []

    # Create a list of valid audio files, using tqdm to show progress
    valid_files = []
    with tqdm(total=total_files, desc="Validating audio files", unit="file") as pbar:
        for file_path in wav_files:
            if is_valid_audio(file_path):
                valid_files.append(file_path)
            pbar.update(1)

    return valid_files

def process_file(file_path, sr_target, noise_level):
    """
    Process a single audio file and return the input and target images.
    """
    # Load the audio file
    try:
      waveform, sr = torchaudio.load(file_path)
      waveform = waveform.cpu().numpy()

      print('File path: ', file_path)

      if sr != sr_target:
          raise ValueError(f"Sample rate mismatch: {sr} != {sr_target}")

      # Convert stereo to mono if needed
      if waveform.shape[0] == 2:
          audio = waveform.mean(axis=0)
      else:
          audio = waveform[0]

      # Add channel dimension
      #audio = audio[np.newaxis, :]

      # Process audio to input and target images
      input_image, target_image = process_audio_to_image(audio, sr, noise_level=noise_level)
      return input_image, target_image
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_checkpoint(checkpoint_file, default_value=0):

  # Step 2: Specify the path to the file using pathlib
  file_path = Path(checkpoint_file)
  print(f"Checkpoint file path: {file_path}")
  # Step 3: Read the content of the file
  try:
      with file_path.open('r') as f:
          # Assuming the file contains a single number (for example: '123')
          checkpoint = int(f.read().strip())  # Convert content to an integer
          print(f"Checkpoint value: {checkpoint}")
          return checkpoint
  except Exception as e:
      print(f"Error reading file: {e}")
      return default_value


  """
    Load the last processed batch index from the checkpoint file stored in Google Drive.

    Args:
        checkpoint_file (str): Path to the checkpoint file (relative to Google Drive).
        default_value (int): Value to return if the file doesn't exist or can't be read.

    Returns:
        int: The last processed batch index, or `default_value` if the checkpoint is unavailable.
    """
    # Mount Google Drive to access its files
#    drive.mount('/content/drive', force_remount=True)

    # Define the path to the file in Google Drive
#    drive_file_path = checkpoint_file  # Adjust path if needed

    # Temporary path in Colab to store the checkpoint file
#    temp_file_path = '/content/' + os.path.basename(checkpoint_file)

#    try:
#        print(f"Checkpoint file found in Google Drive: {drive_file_path}")
        # Step 2: Download the file to Colab's temporary storage
#        shutil.copy(drive_file_path, temp_file_path)

        # Step 3: Read the file from the temporary storage
#        print(f"Reading the checkpoint file from temporary storage: {temp_file_path}")
#        with open(temp_file_path, 'r') as f:
#            return int(f.read().strip())
#    except (OSError, ValueError) as e:
#        print(f"Error reading the checkpoint file from temporary storage: {e}. Returning default value {default_value}.")
#    else:
#        print(f"Checkpoint file {drive_file_path} not found in Google Drive. Returning default value {default_value}.")

#    return default_value



def save_checkpoint(checkpoint_file, batch_index):
    """Save the last processed batch index to the checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        f.write(str(batch_index))

def get_all_wav_files(data_dir, batch_size=100):
    all_wav_files = []
    gathered = False
    while not gathered:
      try:
          i = 0
          with os.scandir(data_dir) as entries:
              #print('files_processed: ', i)
              batch = []
              for entry in entries:
                  if entry.is_file() and entry.name.endswith('.wav'):
                      i += 1
                      batch.append(entry.path)
                      if len(batch) == batch_size:
                          all_wav_files.extend(batch)  # Add batch to final list
                          batch = []  # Reset batch

              # Add any remaining files in the last batch
              if batch:
                  all_wav_files.extend(batch)

          print('Files processed: ', len(all_wav_files))
          gathered = True
          return all_wav_files

      except OSError as e:
          print(f"Error accessing directory: {e}")

def process_and_save_dataset(data_dir, output_file, checkpoint_file, sr_target=44100, noise_level=0.001, batch_size=10, max_length=1024):
    """
    Process all .wav files in a folder and save the input-output image pairs.
    Use this to train an autoencoder (with a small amount of added noise)
    This should be used within an if __name__ == '__main__' function for speed
    """
    # Get the list of .wav files in the directory
    print('Gathering wav files....')
    wav_files = get_all_wav_files(data_dir)
    total_files = len(wav_files)
    if total_files == 0:
        print("No .wav files found in the specified directory.")
        return

    # Load checkpoint to resume processing
    start_batch_idx = load_checkpoint(checkpoint_file)
    print(f"Resuming from batch index: {start_batch_idx}")

    # Create HDF5 file for saving
    print('Creating HDF5 file....')
    with h5py.File(output_file, 'a') as h5f:
        if "input_images" not in h5f:
          input_dataset = h5f.create_dataset(
              "input_images",
              shape=(0, 3, 1024, 86),  # Initially empty along the first dimension
              maxshape=(None, 3, 1024, 86),  # Unlimited along the first dimension
              dtype=np.float32,
              compression="gzip"
          )
        else:
          input_dataset = h5f["input_images"]

        if "target_images" not in h5f:
          target_dataset = h5f.create_dataset(
              "target_images",
              shape=(0, 3, 1024, 86),  # Initially empty along the first dimension
              maxshape=(None, 3, 1024, 86),  # Unlimited along the first dimension
              dtype=np.float32,
              compression="gzip"
          )
        else:
          target_dataset = h5f["target_images"]

        # Process in batches
        for i in tqdm(range(start_batch_idx, total_files, batch_size), desc="Processing batches", unit="batch"):
            batch_files = wav_files[i:i + batch_size]
            input_images = []
            target_images = []

            # Process files in parallel
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_file, file, sr_target, noise_level) for file in batch_files]
                results = [future.result() for future in futures if future.result() is not None]

            # Collect batch results
            for input_image, target_image in results:
                input_images.append(input_image)
                target_images.append(target_image)

            print(input_images[0].shape)
            print(target_images[0].shape)

            # Append batch to datasets
            input_dataset.resize(input_dataset.shape[0] + len(input_images), axis=0)
            target_dataset.resize(target_dataset.shape[0] + len(target_images), axis=0)
            input_dataset[-len(input_images):] = np.stack(input_images)
            target_dataset[-len(target_images):] = np.stack(target_images)

            # Save checkpoint after each batch
            save_checkpoint(checkpoint_file, i + batch_size)

    print(f"Dataset saved to {output_file}")

  # Usage example

import numpy as np

def calculate_snr(signal, noise):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        signal (numpy.ndarray): The clean signal array.
        noise (numpy.ndarray): The noise array.
        
    Returns:
        float: The calculated SNR in decibels (dB).
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def combine_signal_noise(signal, noise, target_snr_db):
    """
    Combine signal and noise at a specific target SNR (dB).
    
    Args:
        signal (numpy.ndarray): The clean signal array.
        noise (numpy.ndarray): The noise array.
        target_snr_db (float): Desired SNR in decibels.
        
    Returns:
        numpy.ndarray: The combined signal with the specified SNR.
    """
    # Calculate current SNR
    current_snr_db = calculate_snr(signal, noise)
    
    # Calculate scaling factor for noise to achieve target SNR
    scaling_factor = 10**((current_snr_db - target_snr_db) / 20)
    
    # Scale noise and combine
    scaled_noise = noise * scaling_factor
    combined_signal = signal + scaled_noise
    
    return combined_signal, current_snr_db

def process_audio_and_noise_to_image(audio, noise, sr, plot=False, random_noise_level=0, background_noise_level=0, SNRdB=None):
  # Parameters
  T = np.linspace(0, len(audio)/sr, len(audio), endpoint=False)
  # --
  if plot:
    plt.plot(audio)
    plt.xlim((4000, 8000))
    plt.show()

  # Add Gaussian noise with random intensity
  sigma = 1  # 1 for pink noise, 0 for white noise, 2 for brown noise
  mu = 0
  s = np.random.normal(mu, sigma, len(audio))* np.sin(2 * np.pi * random.uniform(0.1, 1.5) * T)
  if not random_noise_level:
    s = (s/np.max(abs(s)))
  elif random_noise_level > 0:
    s = (s/np.max(abs(s))) * random.uniform(0, random_noise_level)
  elif random_noise_level < 0:
    s = (s/np.max(abs(s))) * (-random_noise_level)

  if plot:
    plt.plot(s)
    plt.xlim((4000, 8000))
    plt.show()

  if background_noise_level > 0:
    noise = (noise/np.max(abs(noise))) * random.uniform(0, background_noise_level)
  else:
    noise = (noise/np.max(abs(noise))) * 0

  if SNRdB is None:
    random_number = random.uniform(0, 1)
    if random_number > 0.7:
      noisy_audio = np.clip(audio + s + noise, -1, 1)
    elif random_number > 0.2:
      noisy_audio = np.clip(audio + noise, -1, 1)
    else:
      noisy_audio = audio

  else:
     noise = noise + s
     target_snr_db = random.uniform(SNRdB, 40)
     noisy_audio, _ = combine_signal_noise(audio, noise, target_snr_db)
     
  if plot:
    plt.plot(noisy_audio)
    plt.xlim((4000, 8000))
    plt.show()

  #noisy_audio = bandpass_filter(noisy_audio, 80, 16000, sr, order=1)

  input_image = audio_to_image(noisy_audio, sr)
  target_image = audio_to_image(audio, sr)

  if plot:
    # Display the array as an image
    plt.subplot(6, 1, 1)
    librosa.display.specshow(input_image[0], sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Magnitude Spectrogram')

    plt.subplot(6, 1, 2)
    librosa.display.specshow(input_image[1], sr=sr, y_axis="log", x_axis='time', cmap='twilight')
    plt.colorbar()
    plt.title('Magnitude Spec')

    plt.subplot(6, 1, 3)
    librosa.display.specshow(input_image[2], sr=sr, y_axis="log", x_axis='time', cmap='twilight')
    plt.colorbar()
    plt.title('Phase Spectrogram')

    plt.subplot(6, 1, 4)
    librosa.display.specshow(target_image[0], sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Magnitude Spectrogram')

    plt.subplot(6, 1, 5)
    librosa.display.specshow(target_image[1], sr=sr, y_axis="log", x_axis='time', cmap='twilight')
    plt.colorbar()
    plt.title('Magnitude Spectrogram')

    plt.subplot(6, 1, 6)
    librosa.display.specshow(target_image[2], sr=sr, y_axis="log", x_axis='time', cmap='twilight')
    plt.colorbar()
    plt.title('Phase Spectrogram')

    #plt.tight_layout()
    plt.show()

  return input_image, target_image

def load_audio_file(file_path):
  waveform, sr = torchaudio.load(file_path)
  waveform = waveform.cpu().numpy()

  if sr != 44100:
      raise ValueError(f"Sample rate mismatch: {sr} != {44100}")

  # Convert stereo to mono if needed
  if waveform.shape[0] == 2:
      audio = waveform.mean(axis=0)
  else:
      audio = waveform[0]

  return audio, sr

def ensure_file_closed(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            print(f"File {file_path} is accessible and will be closed.")
    except OSError as e:
        print(f"Could not access {file_path}: {e}")

def close_file_handles(file_path):
    current_process = psutil.Process(os.getpid())
    for handle in current_process.open_files():
        if file_path in handle.path:
            print(f"Closing file handle: {handle.path}")
            os.close(handle.fd)

def process_file(file_path, noise_file, background_noise_level, random_noise_level, SNRdB):
    """
    Process a single audio file and return the input and target images.
    """
    # Load the audio file
    try:

      audio, audio_sr = load_audio_file(file_path)
      noise, noise_sr = load_audio_file(noise_file)

      assert(audio_sr == noise_sr)
      sr = audio_sr
      # Add channel dimension


      # Process audio to input and target images
      input_image, target_image = process_audio_and_noise_to_image(audio, noise, sr, background_noise_level=background_noise_level, random_noise_level=random_noise_level, SNRdB=SNRdB)
      return input_image, target_image
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def ensure_directory_exists(file_path):
    """
    Creates all directories in the provided file path if they do not exist.

    Parameters:
    - file_path (str): The full file path where directories need to be created.

    Returns:
    None
    """
    # Extract the directory part of the file path
    directory = os.path.dirname(file_path)

    # Check if the directory exists, and create it if it doesn't
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def check_file_exists(filepath):
    """
    Checks if a file exists at the given path and prints a statement.

    Parameters:
        filepath (str): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if os.path.exists(filepath):
        print(f"The file '{filepath}' exists.")
        return True
    else:
        print(f"ERROR! The file '{filepath}' does not exist.")
        return False

import traceback

def process_and_save_noisy_dataset(
      data_dir, 
      noise_data_dir, 
      output_file, 
      checkpoint_file, 
      sr_target=44100, 
      background_noise_level=0.01, 
      random_noise_level=0.001, 
      batch_size=10, 
      max_length=1024, 
      process_pool=True, 
      manual_checkpoint=None, 
      SNRdB=None, 
      verbose=False, 
      checkpoint_file_size=100, 
      max_file_size_gb=60):
    """
    Process all .wav files in a folder and save the input-output image pairs.
    use this to train a denoising autoencoder 
    This should be used within an if __name__ == '__main__' function for speed
    """
    try:
      random.seed(42)
      LOGIC = True
      max_file_size_bytes = max_file_size_gb * 1024**3

      while LOGIC:
        # Get the list of .wav files in the directory
        print('Gathering wav files....')
        wav_files = get_all_wav_files(data_dir)
        random.shuffle(wav_files)
        total_files = len(wav_files)
        if total_files == 0:
            print("No .wav files found in the specified directory.")
            return

        # get noise files
        print('Gathering noise files....')
        noise_wav_files = get_all_wav_files(noise_data_dir)
        random.shuffle(noise_wav_files)
        total_noise_files = len(noise_wav_files)
        if total_noise_files == 0:
            print("No .wav files found in the specified directory.")
            return

        # Load checkpoint to resume processing
        if manual_checkpoint is not None:
          start_batch_idx = manual_checkpoint
        else:
          start_batch_idx = load_checkpoint(checkpoint_file)
        print(f"Resuming from batch index: {start_batch_idx}")

        # Create HDF5 file for saving
        print('Creating HDF5 file....')
        with h5py.File(output_file, 'a') as h5f:
            if "input_images" not in h5f:
              input_dataset = h5f.create_dataset(
                  "input_images",
                  shape=(0, 3, 1025, 89),  # Initially empty along the first dimension
                  maxshape=(None, 3, 1025, 89),  # Unlimited along the first dimension
                  dtype=np.float32,
                  compression="gzip"
              )
            else:
              input_dataset = h5f["input_images"]

            if "target_images" not in h5f:
              target_dataset = h5f.create_dataset(
                  "target_images",
                  shape=(0, 3, 1025, 89),  # Initially empty along the first dimension
                  maxshape=(None, 3, 1025, 89),  # Unlimited along the first dimension
                  dtype=np.float32,
                  compression="gzip"
              )
            else:
              target_dataset = h5f["target_images"]

            print('checking for file existance....')
            check_file_exists(output_file)
            print('processing batches...')
            # Process in batches
            for i in tqdm(range(start_batch_idx, total_files, batch_size), desc="Processing batches", unit="batch"):
                batch_files = wav_files[i:i + batch_size]
                noise_files = random.sample(noise_wav_files, batch_size)

                # Initialize lists to store input and target images for the batch
                input_images = []
                target_images = []

                #print(np.array([batch_files, noise_files]).T[:5])

                # Process files in parallel
                if process_pool:
                  with ProcessPoolExecutor() as executor:
                      futures = [
                        executor.submit(process_file, audio_file, noise_file, background_noise_level, random_noise_level, SNRdB) 
                        for audio_file, noise_file in zip(batch_files, noise_files)
                        ]
                      results = [future.result() for future in futures if future.result() is not None]

                  # Collect batch results
                  for input_image, target_image in results:
                      input_images.append(input_image)
                      target_images.append(target_image)

                # process files as a loop
                else:
                  for audio_file, noise_file in zip(batch_files, noise_files):
                    output = process_file(audio_file, noise_file, background_noise_level, random_noise_level, SNRdB)

                    # collect batch results
                    if output is not None:
                      input_image, target_image = output
                      input_images.append(input_image)
                      target_images.append(target_image)
                    else:
                      print(f"Error processing {audio_file}")

                # Append batch to datasets
                input_dataset.resize(input_dataset.shape[0] + len(input_images), axis=0)
                target_dataset.resize(target_dataset.shape[0] + len(target_images), axis=0)
                input_dataset[-len(input_images):] = np.stack(input_images)
                target_dataset[-len(target_images):] = np.stack(target_images)

                if verbose:
                  print('input dataset shape:', input_dataset.shape)
                  print('target dataset shape:', target_dataset.shape)
                # Save checkpoint after each batch
                save_checkpoint(checkpoint_file, i + batch_size)

                if os.path.exists(output_file):
                  current_size = os.path.getsize(output_file)
                  if current_size >= max_file_size_bytes:
                    LOGIC = False

                if i > checkpoint_file_size:
                  break

    except Exception as e:
        # Log the exception
        print("An error occurred:")
        traceback.print_exc()
    finally:
        # Ensure the HDF5 file is flushed and closed properly
        print("Processing complete. HDF5 file saved and closed.")

    print(f"Dataset saved to {output_file}")

# Function to calculate RMS of an audio signal
def calculate_rms(audio):
    return np.sqrt(np.mean(audio**2))

# generate shorter audio files from audio files
def generate_audio_files(input_path, output_path, t=1, sr=44100):
    '''
    Generate audio dataset of a certain length within an output path from an input path 
    could be improved by pool processing
    '''
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # List all the .wav files to count total files for progress bar
    all_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                all_files.append(os.path.join(root, file))

    # Set up tqdm progress bar for the files
    with tqdm(total=len(all_files), desc="Processing files") as file_bar:
        # Iterate over the files
        for file in all_files:
            # Load audio file using librosa
            audio, sr = librosa.load(file, sr=sr)  # 44.1kHz resampling
            print(f'sr: {sr}')
            print(f'audio_shape: {np.shape(audio)}')

            # Get the length of the audio in seconds
            total_duration = librosa.get_duration(y=audio, sr=sr)

            # Create a unique filename using the full path (without the base input path)
            relative_path = os.path.relpath(file, input_path).replace(os.sep, "_")  # Replace path separators with underscores
            base_filename = os.path.splitext(relative_path)[0]  # Remove the file extension for uniqueness

            # Split the audio into t-second chunks and update progress bar
            for start_time in range(0, int(total_duration), t):
                # Create the output filename
                output_filename = f"{base_filename}_sec{start_time+t}.wav"
                output_file_path = os.path.join(output_path, output_filename)

                # Check if the file already exists
                if not os.path.exists(output_file_path):
                    start_sample = int(start_time * sr)
                    end_sample = int((start_time + t) * sr)
                    audio_chunk = audio[start_sample:end_sample]

                    # Save the chunk as a new .wav file
                    if calculate_rms(audio_chunk) > 0.015: # 0.02 if music
                        sf.write(output_file_path, audio_chunk, sr)

            # Update file progress
            file_bar.update(1)

    print("Audio files have been split and saved.")

## -----
if __name__ == '__main__':
    # Training folders
    train_noise_data_dir = '/content/drive/MyDrive/Datasets/Noise/train-1s-44-1khz/'
    train_data_dir = '/content/drive/MyDrive/Datasets/Music/MUSDB18/train-1s-44-1khz/'  # Replace with the path to your .wav files

    train_output_file = "/content/drive/MyDrive/Datasets/Music-Noise/train-1s-44-1khz-magnitude-freqweightmagnitude-phase.h5"  # Replace with the desired output path
    train_checkpoint_file = "/content/drive/MyDrive/Datasets/Music-Noise/train-1s-44-1khz-magnitude-freqweightmagnitude-phase___checkpoint.txt"
    #manual_train_checkpoint_number = 73900

    # Testing folders
    test_noise_data_dir = '/content/drive/MyDrive/Datasets/Noise/test-1s-44-1khz/'
    test_data_dir = '/content/drive/MyDrive/Datasets/Music/MUSDB18/test-1s-44-1khz/'  # Replace with the path to your .wav files

    test_output_file = "/content/drive/MyDrive/Datasets/Music-Noise/test-1s-44-1khz-magnitude-freqweightmagnitude-phase.h5"  # Replace with the desired output path
    test_checkpoint_file = "/content/drive/MyDrive/Datasets/Music-Noise/test-1s-44-1khz-magnitude-freqweightmagnitude-phase___checkpoint.txt"

    batch_size = 150
    noise_level = 0.05
    process_pool = True

    PROCESS_TRAIN = False
    PROCESS_TEST = False

    # make sure h5 files are closed
    ensure_file_closed(train_output_file)
    ensure_file_closed(test_output_file)
    #close_file_handles(train_output_file)
    #close_file_handles(test_output_file)
    print('closed files...')

    # Process and save dataset
    if PROCESS_TRAIN:
        print('Processing training dataset....')
        process_and_save_noisy_dataset(train_data_dir, train_noise_data_dir, train_output_file, train_checkpoint_file, batch_size=batch_size, noise_level=noise_level, process_pool=process_pool)

    if PROCESS_TEST:
        print('Processing testing dataset....')
        process_and_save_noisy_dataset(test_data_dir, test_noise_data_dir, test_output_file, test_checkpoint_file, batch_size=batch_size, noise_level=noise_level, process_pool=process_pool)

    test_data_dir = '/content/drive/MyDrive/Datasets/Music/MUSDB18/test-1s-44-1khz/'  # Replace with the path to your .wav files
    test_output_file = "/content/drive/MyDrive/Datasets/Music/MUSDB18/test-1s-44-1khz-magnitude-freqweightmagnitude-phase.h5"  # Replace with the desired output path
    test_checkpoint_file = "/content/drive/MyDrive/Datasets/Music/MUSDB18/test-1s-44-1khz-magnitude-freqweightmagnitude-phase___checkpoint.txt"

    train_data_dir = '/content/drive/MyDrive/Datasets/Music/MUSDB18/train-1s-44-1khz/'  # Replace with the path to your .wav files
    train_output_file = "/content/drive/MyDrive/Datasets/Music/MUSDB18/train-1s-44-1khz-magnitude-freqweightmagnitude-phase.h5"  # Replace with the desired output path
    train_checkpoint_file = "/content/drive/MyDrive/Datasets/Music/MUSDB18/train-1s-44-1khz-magnitude-freqweightmagnitude-phase___checkpoint.txt"

    batch_size = 100

    PROCESS_TRAIN = False
    PROCESS_TEST = False

    # Process and save dataset
    if PROCESS_TRAIN:
        print('Processing training dataset....')
        process_and_save_dataset(train_data_dir, train_output_file, train_checkpoint_file, batch_size=batch_size)

    if PROCESS_TEST:
        print('Processing testing dataset....')
        process_and_save_dataset(test_data_dir, test_output_file, test_checkpoint_file, batch_size=batch_size)
