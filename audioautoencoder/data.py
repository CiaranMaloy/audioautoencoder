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
    
    return combined_signal, scaled_noise, current_snr_db

def process_audio_extract_features(audio, noise, sr, plot=False, random_noise_level=0, background_noise_level=0, SNRdB=None, audio_length=44100):
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
    noise = (noise/np.max(abs(noise))) * 0.00001

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
     target_snr_db = random.uniform(SNRdB[0], SNRdB[1])
     noisy_audio, scaled_noise, _ = combine_signal_noise(audio, noise, target_snr_db)
     
  if plot:
    plt.plot(noisy_audio)
    plt.xlim((4000, 8000))
    plt.show()

  ## Scale and normalise by level - to decouple snr from overall signal level
  normalisation_factor = np.max(np.stack((noisy_audio, audio, scaled_noise)))
  # devide by max value - make max 1
  noisy_audio, audio, scaled_noise = noisy_audio/normalisation_factor, audio/normalisation_factor, scaled_noise/normalisation_factor
  # randomise signal level between 0 and -6 dB
  signal_level_db = random.uniform(-6, 0)
  signal_level_amp = 10**((signal_level_db) / 20)
  noisy_audio, audio, scaled_noise = noisy_audio*signal_level_amp, audio*signal_level_amp, scaled_noise*signal_level_amp

  # extract features
  input_features = extract_features(noisy_audio, sr, audio_length=audio_length)
  target_features = extract_features(audio, sr, audio_length=audio_length)
  noise_features = extract_features(scaled_noise, sr, audio_length=audio_length)

  return input_features, target_features, noise_features, target_snr_db

def process_audio_separation_to_image(audio, noise, sr, plot=False, random_noise_level=0, background_noise_level=0, SNRdB=None, audio_length=44100):
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
    noise = (noise/np.max(abs(noise))) * 0.00001

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
     target_snr_db = random.uniform(SNRdB[0], SNRdB[1])
     noisy_audio, scaled_noise, _ = combine_signal_noise(audio, noise, target_snr_db)
     
  if plot:
    plt.plot(noisy_audio)
    plt.xlim((4000, 8000))
    plt.show()

  #noisy_audio = bandpass_filter(noisy_audio, 80, 16000, sr, order=1)

  # extract features
  extract_features(audio, sr, audio_length=44100*2)

  input_image = audio_to_image(noisy_audio, sr, audio_length=audio_length, features=False)
  target_image = audio_to_image(audio, sr, audio_length=audio_length, features=False)
  noise_image = audio_to_image(scaled_noise, sr, audio_length=audio_length, features=False)

  return input_image, target_image, noise_image, target_snr_db

def process_audio_and_noise_to_image(audio, noise, sr, plot=False, random_noise_level=0, background_noise_level=0, SNRdB=None, audio_length=44100):
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
     target_snr_db = random.uniform(SNRdB[0], SNRdB[1])
     noisy_audio, _ = combine_signal_noise(audio, noise, target_snr_db)
     
  if plot:
    plt.plot(noisy_audio)
    plt.xlim((4000, 8000))
    plt.show()

  #noisy_audio = bandpass_filter(noisy_audio, 80, 16000, sr, order=1)

  input_image = audio_to_image(noisy_audio, sr, audio_length=audio_length)
  target_image = audio_to_image(audio, sr, audio_length=audio_length)

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

import torchaudio.transforms as T

def load_audio_file(file_path, target_sr=44100):
  waveform, sr = torchaudio.load(file_path)
  waveform = waveform.cpu().numpy()

  # Resample if needed
  if sr != target_sr:
      resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
      waveform = resampler(waveform)
      sr = target_sr  # Update sample rate

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

def process_file(file_path, noise_file, background_noise_level, random_noise_level, SNRdB, audio_length):
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
      input_features, target_features, noise_features, target_snr_db = process_audio_extract_features(audio, noise, sr, background_noise_level=background_noise_level, random_noise_level=random_noise_level, SNRdB=SNRdB, audio_length=audio_length)
      return input_features, target_features, noise_features, file_path, noise_file, target_snr_db
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

# Function to calculate RMS of an audio signal
def calculate_rms(audio):
    return np.sqrt(np.mean(audio**2))

# generate shorter audio files from audio files
def generate_audio_files(input_path, output_path, t=1, sr=44100, min_size=0.005):
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
            #print(f'sr: {sr}')
            #print(f'audio_shape: {np.shape(audio)}')

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
                    if calculate_rms(audio_chunk) > min_size: # 0.02 if music
                        if len(audio_chunk) == int(t * sr):
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
