import time
import random
import traceback
from threading import Lock

from data import *

from datetime import datetime
import os

def add_datetime_to_filename(filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Get the current datetime
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Combine the name, datetime, and extension
    unique_filename = f"{name}_{current_datetime}{ext}"
    return unique_filename

def gather_wav_files_and_report(data_dir):
    wav_files = get_all_wav_files(data_dir)
    random.shuffle(wav_files)
    total_files = len(wav_files)
    if total_files == 0:
        print("No .wav files found in the specified directory.")
        return
    return wav_files, total_files

def create_dataset(dataset_name, h5f, c=3, h=1025, w=89):
    if dataset_name not in h5f:
        dataset = h5f.create_dataset(
            dataset_name,
            shape=(0, c, h, w),  # Initially empty along the first dimension
            maxshape=(None, c, h, w),  # Unlimited along the first dimension
            dtype=np.float32,
            compression="gzip"
        )
    else:
        dataset = h5f[dataset_name]
    
    return dataset

def process_and_save_noisy_dataset(
      data_dir, 
      noise_data_dir, 
      output_file, 
      checkpoint_file, 
      audio_length=44100,
      background_noise_level=0.01, 
      random_noise_level=0.001, 
      batch_size=10, 
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
      write_lock = Lock()

      while LOGIC:
        # Get the list of .wav files in the directory
        print('Gathering wav files....')
        wav_files, total_files = gather_wav_files_and_report(data_dir)

        # get noise files
        print('Gathering noise files....')
        noise_wav_files, _ = gather_wav_files_and_report(noise_data_dir)

        # Load checkpoint to resume processing
        if manual_checkpoint is not None:
          start_batch_idx = manual_checkpoint
        else:
          start_batch_idx = load_checkpoint(checkpoint_file)
        print(f"Resuming from batch index: {start_batch_idx}")

        # 
        for i in tqdm(range(start_batch_idx, total_files, batch_size), desc="Processing batches", unit="batch"):
            batch_files = wav_files[i:i + batch_size]
            noise_files = random.sample(noise_wav_files, batch_size)

            # Initialize lists to store input and target images for the batch
            input_images = []
            target_images = []

            # Process files in parallel
            if process_pool:
                with ProcessPoolExecutor() as executor:
                    futures = [
                    executor.submit(process_file, audio_file, noise_file, background_noise_level, random_noise_level, SNRdB, audio_length) 
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
                    output = process_file(audio_file, noise_file, background_noise_level, random_noise_level, SNRdB, audio_length)

                # collect batch results
                if output is not None:
                    input_image, target_image = output
                    input_images.append(input_image)
                    target_images.append(target_image)
                else:
                    print(f"Error processing {audio_file}")

            # convert to numpy arrays
            input_images = np.array(input_images, dtype=np.float32)
            target_images = np.array(target_images, dtype=np.float32)

            # raise error if is inf or nan
            if np.isnan(np.stack(input_images)).any() or np.isinf(np.stack(input_images)).any():
                raise ValueError("Input images contain NaN or Inf values.")

                    # Create HDF5 file for saving
            print('Creating HDF5 file....')
            sub_output_file = add_datetime_to_filename(output_file)

            with h5py.File(sub_output_file, 'a') as h5f:
                input_dataset = create_dataset("input_images", h5f)
                target_dataset = create_dataset("target_images", h5f)

                print('checking for file existance....')
                check_file_exists(sub_output_file)

                print('processing batches...')
                # Process in batches
                # Append batch to datasets
                with write_lock:
                  input_dataset.resize(input_dataset.shape[0] + len(input_images), axis=0)
                  target_dataset.resize(target_dataset.shape[0] + len(target_images), axis=0)
                  input_dataset[-len(input_images):] = np.stack(input_images)
                  target_dataset[-len(target_images):] = np.stack(target_images)

                # dedicated write to file
                h5f.flush()

                if verbose:
                  print('input dataset shape:', input_dataset.shape)
                  print('target dataset shape:', target_dataset.shape)
                # Save checkpoint after each batch
                save_checkpoint(checkpoint_file, i + batch_size)

                if os.path.exists(sub_output_file):
                  current_size = os.path.getsize(sub_output_file)
                  #i.set_postfix(loss=f"{current_size / 1024**3}")
                  if current_size >= max_file_size_bytes:
                    LOGIC = False
                    print('File maximum size met....')
                    time.sleep(20)
                    break

                if i > checkpoint_file_size:
                  LOGIC = False
                  print('File maximum samples file......')
                  time.sleep(20)
                  break

            if os.path.exists(sub_output_file):
                current_size = os.path.getsize(sub_output_file)
                print(f'Current file size: {current_size / 1024**3}')
            print(f'Done {sub_output_file}')

    except Exception as e:
        # Log the exception
        print("An error occurred:")
        traceback.print_exc()
    finally:
        # Ensure the HDF5 file is flushed and closed properly
        print("Processing complete. HDF5 file saved and closed.")

    print(f"Dataset saved to {output_file}")

def process_and_save_separation_dataset(
      data_dir, 
      noise_data_dir, 
      output_file, 
      checkpoint_file, 
      audio_length=44100,
      background_noise_level=0.01, 
      random_noise_level=0.001, 
      batch_size=500, 
      process_pool=True, 
      manual_checkpoint=None, 
      SNRdB=None, 
      verbose=False, 
      checkpoint_file_size=50000, 
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
      write_lock = Lock()

      while LOGIC:
        # Get the list of .wav files in the directory
        print('Gathering wav files....')
        wav_files, total_files = gather_wav_files_and_report(data_dir)

        # get noise files
        print('Gathering noise files....')
        noise_wav_files, _ = gather_wav_files_and_report(noise_data_dir)

        # Load checkpoint to resume processing
        if manual_checkpoint is not None:
          start_batch_idx = manual_checkpoint
        else:
          start_batch_idx = load_checkpoint(checkpoint_file)
        print(f"Resuming from batch index: {start_batch_idx}")

        # 
        for i in tqdm(range(start_batch_idx, total_files, batch_size), desc="Processing batches", unit="batch"):
            batch_files = wav_files[i:i + batch_size]
            noise_files = random.sample(noise_wav_files, batch_size)

            # Initialize lists to store input and target images for the batch
            input_images = []
            target_images = []

            # Process files in parallel
            if process_pool:
                with ProcessPoolExecutor() as executor:
                    futures = [
                    executor.submit(process_file, audio_file, noise_file, background_noise_level, random_noise_level, SNRdB, audio_length) 
                    for audio_file, noise_file in zip(batch_files, noise_files)
                    ]
                    results = [future.result() for future in futures if future.result() is not None]

                # Collect batch results
                for input_image, target_image, noise_image in results:
                    #print(np.shape(input_image))
                    #print(np.shape([target_image[0], noise_image[0]]))
                    input_images.append(input_image)
                    target_images.append([target_image[0], noise_image[0]])


            # process files as a loop
            else:
                for audio_file, noise_file in zip(batch_files, noise_files):
                    output = process_file(audio_file, noise_file, background_noise_level, random_noise_level, SNRdB, audio_length)

                # collect batch results
                if output is not None:
                    input_image, target_image, noise_image = output
                    input_images.append(input_image)
                    target_images.append([target_image[0], noise_image[0]])
                else:
                    print(f"Error processing {audio_file}")

            # convert to numpy arrays
            input_images = np.array(input_images, dtype=np.float32)
            target_images = np.array(target_images, dtype=np.float32)

            # raise error if is inf or nan
            if np.isnan(np.stack(input_images)).any() or np.isinf(np.stack(input_images)).any():
                raise ValueError("Input images contain NaN or Inf values.")

                    # Create HDF5 file for saving
            print('Creating HDF5 file....')
            sub_output_file = add_datetime_to_filename(output_file)

            with h5py.File(sub_output_file, 'a') as h5f:
                input_dataset = create_dataset("input_images", h5f, c=2, w=175)
                target_dataset = create_dataset("target_images", h5f, c=2, w=175)

                print('checking for file existance....')
                check_file_exists(sub_output_file)

                print('processing batches...')
                # Process in batches
                # Append batch to datasets
                with write_lock:
                  input_dataset.resize(input_dataset.shape[0] + len(input_images), axis=0)
                  target_dataset.resize(target_dataset.shape[0] + len(target_images), axis=0)
                  input_dataset[-len(input_images):] = np.stack(input_images)
                  target_dataset[-len(target_images):] = np.stack(target_images)

                # dedicated write to file
                h5f.flush()

                if verbose:
                  print('input dataset shape:', input_dataset.shape)
                  print('target dataset shape:', target_dataset.shape)
                # Save checkpoint after each batch
                save_checkpoint(checkpoint_file, i + batch_size)

                if os.path.exists(sub_output_file):
                  current_size = os.path.getsize(sub_output_file)
                  #i.set_postfix(loss=f"{current_size / 1024**3}")
                  if current_size >= max_file_size_bytes:
                    LOGIC = False
                    print('File maximum size met....')
                    time.sleep(20)
                    break

                if i > checkpoint_file_size:
                  LOGIC = False
                  print('File maximum samples file......')
                  time.sleep(20)
                  break

            if os.path.exists(sub_output_file):
                current_size = os.path.getsize(sub_output_file)
                print(f'Current file size: {current_size / 1024**3}')
            print(f'Done {sub_output_file}')

    except Exception as e:
        # Log the exception
        print("An error occurred:")
        traceback.print_exc()
    finally:
        # Ensure the HDF5 file is flushed and closed properly
        print("Processing complete. HDF5 file saved and closed.")

    print(f"Dataset saved to {output_file}")


import h5py
import os
import numpy as np
import math

def combine_h5_files(h5_folder_path, output_folder_path, max_file_size_gb=1, chunk_size=128):
    # Convert max file size to bytes
    max_file_size_bytes = max_file_size_gb * 1024**3
    
    # List all HDF5 files in the folder
    h5_files = [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.endswith(".h5")]
    
    # Open the first file to get the dataset shape
    with h5py.File(h5_files[0], "r") as first_file:
        input_shape = first_file["input_images"].shape[1:]  # Exclude the batch dimension
        target_shape = first_file["target_images"].shape[1:]
        sample_size_bytes = (
            np.prod(input_shape) * np.dtype("float32").itemsize +
            np.prod(target_shape) * np.dtype("float32").itemsize
        )
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Variables for tracking file splitting
    current_file_index = 0
    current_file_samples = 0
    current_file_size = 0
    previous_size = 0
    combined_file = None
    input_dataset = None
    target_dataset = None
    flag = False
    
    def create_new_file():
        """Helper function to create a new HDF5 file."""
        nonlocal current_file_index, current_file_samples, current_file_size, combined_file, input_dataset, target_dataset, previous_size, chunk_size
        # Close the current file if open
        if combined_file is not None:
            combined_file.close()
        
        # Create a new file
        file_path = os.path.join(output_folder_path, f"combined_{current_file_index:03d}.h5")
        combined_file = h5py.File(file_path, "w")
        input_dataset = combined_file.create_dataset("input_images", shape=(0, *input_shape), chunks=(chunk_size, *input_shape), maxshape=(None, *input_shape), dtype="float32")
        target_dataset = combined_file.create_dataset("target_images", shape=(0, *target_shape), chunks=(chunk_size, *input_shape), maxshape=(None, *target_shape), dtype="float32")
        current_file_samples = 0
        current_file_size = 0
        previous_size = 0
        current_file_index += 1
        print(f"Created new file: {file_path}")
    
    # Start with the first file
    create_new_file()
    
    # Copy data from each file into the combined datasets
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as source_file:
            input_data = source_file["input_images"][:]
            target_data = source_file["target_images"][:]
            num_samples = input_data.shape[0]
            
            for i in range(num_samples):
                sample_input = input_data[i:i+1]  # Get one sample at a time
                sample_target = target_data[i:i+1]
                sample_size = sample_size_bytes
                
                # Check if adding this sample exceeds the max file size
                if current_file_size + sample_size > max_file_size_bytes:
                    create_new_file()
                    print('Done....')
                    flag = True
                    break
                
                current_size_gb = (current_file_size + sample_size)/1024**3
                if math.floor(previous_size) != math.floor(current_size_gb):
                    print(np.round(current_size_gb), h5_file)
                previous_size = current_size_gb
                # Append the sample to the current dataset
                input_dataset.resize((current_file_samples + 1, *input_shape))
                target_dataset.resize((current_file_samples + 1, *target_shape))
                input_dataset[current_file_samples] = sample_input
                target_dataset[current_file_samples] = sample_target
                
                current_file_samples += 1
                current_file_size += sample_size

        # Check if adding this sample exceeds the max file size
        if flag:
            print('Done.... - no more files')
            break
    
    # Close the last file
    if combined_file is not None:
        combined_file.close()
    
    print(f"Finished combining files into {current_file_index} output files in {output_folder_path}")

# Example usage
#combine_h5_files("path/to/h5_folder", "path/to/output_folder", max_file_size_gb=1)
