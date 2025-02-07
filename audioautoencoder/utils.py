import os
import torchaudio
import torchaudio.transforms as T
from datetime import datetime

def make_folder(file_path):
    # Check if the directory exists
    directory = os.path.dirname(file_path)

    # If the directory doesn't exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

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

def add_datetime_to_filename(filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Get the current datetime
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Combine the name, datetime, and extension
    unique_filename = f"{name}_{current_datetime}{ext}"
    return unique_filename