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

def create_dataset(dataset_name, h5f):
    if dataset_name not in h5f:
        dataset = h5f.create_dataset(
            dataset_name,
            shape=(0, 3, 1025, 89),  # Initially empty along the first dimension
            maxshape=(None, 3, 1025, 89),  # Unlimited along the first dimension
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

                if os.path.exists(sub_output_file):
                    current_size = os.path.getsize(sub_output_file)
                    print(f'Current file size: {current_size / 1024**3}')

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

    except Exception as e:
        # Log the exception
        print("An error occurred:")
        traceback.print_exc()
    finally:
        # Ensure the HDF5 file is flushed and closed properly
        print("Processing complete. HDF5 file saved and closed.")

    print(f"Dataset saved to {output_file}")