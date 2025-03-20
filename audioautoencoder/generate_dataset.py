import time
import random
import traceback
from threading import Lock

from data import *
from utils import *

import os

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

import os

def ensure_folder(path):
    """
    Ensures that the given path exists. 
    If the path is a file, it creates the parent directory. 
    If the path is a directory, it ensures it exists.
    
    Args:
        path (str): The file or directory path.
    
    Returns:
        str: The ensured directory path.
    """
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

import pandas as pd
import gc
import shutil

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
      max_file_size_gb=60, 
      mix_only=False, 
      folder="/content/", 
      min_free_gb=20):
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

      ensure_folder(output_file)

      while LOGIC:
        # Get the list of .wav files in the directory
        print('Gathering wav files....')
        wav_files, total_files = gather_wav_files_and_report(data_dir)

        if mix_only:
            wav_files = [file for file in wav_files if 'mixture' in file]
            total_files = len(wav_files)
            print('Mix files:', total_files)

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
            # features
            input_features_array, target_features_array, noise_features_array = [], [], []

            # metadata
            filenames = []
            snr_db = []

            # Process files in parallel
            if process_pool:
                with ProcessPoolExecutor() as executor:
                    futures = [
                    executor.submit(process_file, audio_file, noise_file, background_noise_level, random_noise_level, SNRdB, audio_length) 
                    for audio_file, noise_file in zip(batch_files, noise_files)
                    ]
                    results = [future.result() for future in futures if future.result() is not None]

                    # ensure executor shutdown 
                    executor.shutdown(wait=True)
                    del executor
                    gc.collect()

                # Collect batch results
                for input_features, target_features, noise_features, file_path, noise_file, target_snr_db in results:
                    # features 
                    input_features_array.append(input_features)
                    target_features_array.append(target_features)
                    noise_features_array.append(noise_features)

                    # metadata 
                    filenames.append([file_path, noise_file]) # this shoudl be a dict
                    snr_db.append(target_snr_db)

            # process files as a loop
            else:
                print('Blank, there is no logic for this....')
                break
                #for audio_file, noise_file in zip(batch_files, noise_files):
                #    output = process_file(audio_file, noise_file, background_noise_level, random_noise_level, SNRdB, audio_length)

                # collect batch results
                #if output is not None:
                #    input_image, target_image, noise_image, file_path, noise_file = output
                #    input_images.append(input_image)
                #    target_images.append([target_image[0], noise_image[0]])
                #    filenames.append([file_path, noise_file])
                #else:
                #    print(f"Error processing {audio_file}")

            # Metadata arrays
            filenames = np.array(filenames, dtype=h5py.string_dtype())
            snr_db = np.array(snr_db, dtype=np.float32)

            # Convert feature lists into DataFrames
            #df_input = pd.DataFrame(input_features_array)
            #df_target = pd.DataFrame(target_features_array)
            #df_noise = pd.DataFrame(noise_features_array)

            #print(df_input.head())

            #for col in df_input.columns:
            #    print(f"Column: {col}, dtype: {df_input[col].dtype}")

                    # Create HDF5 file for saving
            print('Creating HDF5 file....')
            sub_output_file = add_datetime_to_filename(output_file)

            with h5py.File(sub_output_file, 'a') as h5f:
                # Save features as HDF5 groups
                # input features
                h5f.create_dataset("input_features_phase", data=np.stack([d["phase"] for d in input_features_array]))
                h5f.create_dataset("input_features_spectrogram", data=np.stack([d["spectrogram"] for d in input_features_array]))
                h5f.create_dataset("input_features_edges", data=np.stack([d["edges"] for d in input_features_array]))
                h5f.create_dataset("input_features_cepstrum", data=np.stack([d["cepstrum"] for d in input_features_array]))
                h5f.create_dataset("input_features_cepstrum_edges", data=np.stack([d["cepstrum_edges"] for d in input_features_array]))

                # target features
                #h5f.create_dataset("target_features_phase", data=np.stack([d["phase"] for d in target_features_array]))
                h5f.create_dataset("target_features_spectrogram", data=np.stack([d["spectrogram"] for d in target_features_array]))
                #h5f.create_dataset("target_features_edges", data=np.stack([d["edges"] for d in target_features_array]))
                #h5f.create_dataset("target_features_mfccs", data=np.stack([d["mfccs"] for d in target_features_array]))
                #h5f.create_dataset("target_features_mfcc_delta", data=np.stack([d["mfcc_delta"] for d in target_features_array]))
                #h5f.create_dataset("target_features_mfcc_delta2", data=np.stack([d["mfcc_delta2"] for d in target_features_array]))

                # noise features
                #h5f.create_dataset("noise_features_phase", data=np.stack([d["phase"] for d in noise_features_array]))
                h5f.create_dataset("noise_features_spectrogram", data=np.stack([d["spectrogram"] for d in noise_features_array]))
                #h5f.create_dataset("noise_features_edges", data=np.stack([d["edges"] for d in noise_features_array]))
                #h5f.create_dataset("noise_features_mfccs", data=np.stack([d["mfccs"] for d in noise_features_array]))
                #h5f.create_dataset("noise_features_mfcc_delta", data=np.stack([d["mfcc_delta"] for d in noise_features_array]))
                #h5f.create_dataset("noise_features_mfcc_delta2", data=np.stack([d["mfcc_delta2"] for d in noise_features_array]))

                # Store metadata separately
                h5f.create_dataset("filenames", data=filenames)
                h5f.create_dataset("snr_db", data=snr_db)

                h5f.flush()
                h5f.close()

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

            if (i + batch_size > checkpoint_file_size) or (i + batch_size > total_files):
                LOGIC = False
                print('File maximum samples file......')
                time.sleep(20)
                break

            if os.path.exists(sub_output_file):
                current_size = os.path.getsize(sub_output_file)
                print(f'Current file size: {current_size / 1024**3}')
            print(f'Done {sub_output_file}')

            # ensure feature arrays deleted
            del input_features_array, target_features_array, noise_features_array, h5f
            gc.collect()

            # check free disk space: 
            total, used, free = shutil.disk_usage(folder)
            free_gb = free / (1024**3)  # Convert bytes to GB

            print(f"Free disk space: {free_gb:.2f} GB")
            
            if free_gb < min_free_gb:
                print(f"Warning: Low disk space ({free_gb:.2f} GB left). Stopping script.")
                os._exit(1)  # Force stop the script

    except Exception as e:
        # Log the exception
        print("An error occurred:")
        traceback.print_exc()
    finally:
        # Ensure the HDF5 file is flushed and closed properly
        print("Processing complete. HDF5 file saved and closed.")

    print(f"Dataset saved to {output_file}")

import os
import h5py
import numpy as np
import math

def combine_h5_files_features(h5_folder_path, output_folder_path, max_file_size_gb=1, chunk_size=128):
    """Combines multiple HDF5 files into a few large ones, ensuring they do not exceed max_file_size_gb."""
    
    # Convert max file size to bytes
    max_file_size_bytes = max_file_size_gb * 1024**3

    # List all HDF5 files in the folder
    h5_files = sorted(
        [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.endswith(".h5")]
    )
    
    if not h5_files:
        print("No HDF5 files found in directory.")
        return

    # Open the first file to get dataset structure
    with h5py.File(h5_files[0], "r") as first_file:

        # features
        # input
        input_phase_shape = first_file["input_features_phase"].shape[1:]
        input_spectrogram_shape = first_file["input_features_spectrogram"].shape[1:]
        input_edges_shape = first_file["input_features_edges"].shape[1:]
        input_cepstrum_shape = first_file["input_features_cepstrum"].shape[1:]
        input_cepstrum_edges_shape = first_file["input_features_cepstrum_edges"].shape[1:]

        # output
        target_spectrogram_shape = first_file["target_features_spectrogram"].shape[1:]
        noise_spectrogram_shape = first_file["noise_features_spectrogram"].shape[1:]

        # metadata
        filename_shape = first_file["filenames"].shape[1:]
        snr_db_shape = first_file["snr_db"].shape[1:]

        # dtypes
        # input
        input_phase_dtype = first_file["input_features_phase"].dtype
        input_spectrogram_dtype = first_file["input_features_spectrogram"].dtype
        input_edges_dtype = first_file["input_features_edges"].dtype
        input_cepstrum_dtype = first_file["input_features_cepstrum"].dtype
        input_cepstrum_edges_dtype = first_file["input_features_cepstrum_edges"].dtype

        # output
        target_spectrogram_dtype = first_file["target_features_spectrogram"].dtype
        noise_spectrogram_dtype = first_file["noise_features_spectrogram"].dtype

        # metadata
        filename_dtype = first_file["filenames"].dtype
        snr_db_dtype = first_file["snr_db"].dtype

        sample_size_bytes = (
            np.prod(input_phase_shape) * np.dtype(input_phase_dtype).itemsize +
            np.prod(input_spectrogram_shape) * np.dtype(input_spectrogram_dtype).itemsize +
            np.prod(input_edges_shape) * np.dtype(input_edges_dtype).itemsize +
            np.prod(input_cepstrum_shape) * np.dtype(input_cepstrum_dtype).itemsize +
            np.prod(input_cepstrum_edges_shape) * np.dtype(input_cepstrum_edges_dtype).itemsize +
            np.prod(target_spectrogram_shape) * np.dtype(target_spectrogram_dtype).itemsize +
            np.prod(noise_spectrogram_shape) * np.dtype(noise_spectrogram_dtype).itemsize +
            np.prod(filename_shape) * np.dtype(filename_dtype).itemsize +
            np.prod(snr_db_shape) * np.dtype(snr_db_dtype).itemsize
        )

    # Ensure output directory exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Variables for file management
    current_file_index = 0
    current_file_samples = 0
    current_file_size = 0
    previous_size = 0
    combined_file = None

    # Declare datasets at module scope
    filename_dataset = snr_db_dataset = None
    noise_spectrogram_dataset = input_phase_dataset = input_spectrogram_dataset = input_edges_dataset = input_cepstrum_dataset = input_cepstrum_edges_dataset = target_spectrogram_dataset = None

    def create_new_file():
        """Creates a new HDF5 output file."""
        nonlocal current_file_index, current_file_samples, current_file_size, combined_file
        nonlocal filename_dataset, snr_db_dataset, previous_size  # FIXED
        nonlocal input_phase_dataset, input_spectrogram_dataset, input_edges_dataset, input_cepstrum_dataset, input_cepstrum_edges_dataset, target_spectrogram_dataset, noise_spectrogram_dataset

        # Close previous file if it exists
        if combined_file is not None:
            combined_file.close()

        # Generate new file name
        file_path = os.path.join(output_folder_path, f"combined_{current_file_index:03d}.h5")
        combined_file = h5py.File(file_path, "w")

        # Input datasets
        input_phase_dataset = combined_file.create_dataset(
            "input_features_phase", shape=(0, *input_phase_shape), chunks=(chunk_size, *input_phase_shape),
            maxshape=(None, *input_phase_shape), dtype=input_phase_dtype
        )

        input_spectrogram_dataset = combined_file.create_dataset(
            "input_features_spectrogram", shape=(0, *input_spectrogram_shape), chunks=(chunk_size, *input_spectrogram_shape),
            maxshape=(None, *input_spectrogram_shape), dtype=input_spectrogram_dtype
        )

        input_edges_dataset = combined_file.create_dataset(
            "input_features_edges", shape=(0, *input_edges_shape), chunks=(chunk_size, *input_edges_shape),
            maxshape=(None, *input_edges_shape), dtype=input_edges_dtype
        )

        input_cepstrum_dataset = combined_file.create_dataset(
            "input_features_cepstrum", shape=(0, *input_cepstrum_shape), chunks=(chunk_size, *input_cepstrum_shape),
            maxshape=(None, *input_cepstrum_shape), dtype=input_cepstrum_dtype
        )

        input_cepstrum_edges_dataset = combined_file.create_dataset(
            "input_features_cepstrum_edges", shape=(0, *input_cepstrum_edges_shape), chunks=(chunk_size, *input_cepstrum_edges_shape),
            maxshape=(None, *input_cepstrum_edges_shape), dtype=input_cepstrum_edges_dtype
        )

        # Output dataset
        target_spectrogram_dataset = combined_file.create_dataset(
            "target_features_spectrogram", shape=(0, *target_spectrogram_shape), chunks=(chunk_size, *target_spectrogram_shape),
            maxshape=(None, *target_spectrogram_shape), dtype=target_spectrogram_dtype
        )
        noise_spectrogram_dataset = combined_file.create_dataset(
            "noise_features_spectrogram", shape=(0, *noise_spectrogram_shape), chunks=(chunk_size, *noise_spectrogram_shape),
            maxshape=(None, *noise_spectrogram_shape), dtype=noise_spectrogram_dtype
        )

        # Metadata datasets
        filename_dataset = combined_file.create_dataset(
            "filenames", shape=(0, *filename_shape), chunks=(chunk_size, *filename_shape),
            maxshape=(None, *filename_shape), dtype=filename_dtype
        )

        snr_db_dataset = combined_file.create_dataset(
            "snr_db", shape=(0, *snr_db_shape), chunks=(chunk_size, *snr_db_shape),
            maxshape=(None, *snr_db_shape), dtype=snr_db_dtype
        )

        current_file_samples = 0
        current_file_size = 0
        previous_size = 0
        current_file_index += 1

        print(f"Created new file: {file_path}")

    # Start processing files
    create_new_file()

    break_trigger = False
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as source_file:
            # input
            input_phase = source_file["input_features_phase"][:]
            input_spectrogram = source_file["input_features_spectrogram"][:]
            input_edges = source_file["input_features_edges"][:]
            input_cepstrum = source_file["input_features_cepstrum"][:]
            input_cepstrum_edges = source_file["input_features_cepstrum_edges"][:]

            # output
            target_spectrogram = source_file["target_features_spectrogram"][:]
            noise_spectrogram = source_file["noise_features_spectrogram"]

            # metadata
            filename = source_file["filenames"][:]
            snr_db = source_file["snr_db"][:]

            # number samples
            num_samples = input_phase.shape[0]

            for i in range(0, num_samples, chunk_size):
                # input
                chunk_input_phase = input_phase[i:i+chunk_size]
                chunk_input_spectrogram = input_spectrogram[i:i+chunk_size]
                chunk_input_edges = input_edges[i:i+chunk_size]
                chunk_input_cepstrum = input_cepstrum[i:i+chunk_size]
                chunk_input_cepstrum_edges = input_cepstrum_edges[i:i+chunk_size]

                # output
                chunk_target_spectrogram = target_spectrogram[i:i+chunk_size]
                chunk_noise_spectrogram = noise_spectrogram[i:i+chunk_size]

                # metadata
                chunk_filename = filename[i:i+chunk_size]
                chunk_snr_db = snr_db[i:i+chunk_size]

                # sample size
                chunk_sample_size = chunk_input_phase.shape[0] * sample_size_bytes

                # Check if new file is needed
                if current_file_size + chunk_sample_size > max_file_size_bytes:
                    break_trigger = True
                    break
                    create_new_file()

                # Resize datasets| hi u farted
                new_size = current_file_samples + chunk_input_phase.shape[0]

                # input
                input_phase_dataset.resize((new_size, *input_phase_shape))
                input_spectrogram_dataset.resize((new_size, *input_spectrogram_shape))
                input_edges_dataset.resize((new_size, *input_edges_shape))
                input_cepstrum_dataset.resize((new_size, *input_cepstrum_shape)) 
                input_cepstrum_edges_dataset.resize((new_size, *input_cepstrum_edges_shape)) 

                # target
                target_spectrogram_dataset.resize((new_size, *target_spectrogram_shape))
                noise_spectrogram_dataset.resize((new_size, *noise_spectrogram_shape))

                # metadata
                filename_dataset.resize((new_size, *filename_shape))
                snr_db_dataset.resize((new_size, *snr_db_shape))

                # Append data
                # input
                input_phase_dataset[current_file_samples:new_size] = chunk_input_phase
                input_spectrogram_dataset[current_file_samples:new_size] = chunk_input_spectrogram
                input_edges_dataset[current_file_samples:new_size] = chunk_input_edges
                input_cepstrum_dataset[current_file_samples:new_size] = chunk_input_cepstrum
                input_cepstrum_edges_dataset[current_file_samples:new_size] = chunk_input_cepstrum_edges

                # target
                target_spectrogram_dataset[current_file_samples:new_size] = chunk_target_spectrogram
                noise_spectrogram_dataset[current_file_samples:new_size] = chunk_noise_spectrogram

                # metadata
                filename_dataset[current_file_samples:new_size] = chunk_filename
                snr_db_dataset[current_file_samples:new_size] = chunk_snr_db

                current_file_samples = new_size
                current_file_size += chunk_sample_size

                # Print progress every ~1GB
                current_size_gb = current_file_size / 1024**3
                if math.floor(previous_size) != math.floor(current_size_gb):
                    print(f"Progress: {np.round(current_size_gb, 2)} GB - Processing {h5_file}")
                previous_size = current_size_gb
        if break_trigger:
            break
    # Close the last output file
    if combined_file is not None:
        combined_file.close()

    print(f"Finished combining files into {current_file_index} output files in {output_folder_path}")

import random
def combine_h5_files_clean(h5_folder_path, output_folder_path, max_file_size_gb=1, chunk_size=128):
    """Combines multiple HDF5 files into a few large ones, ensuring they do not exceed max_file_size_gb."""
    
    # Convert max file size to bytes
    max_file_size_bytes = max_file_size_gb * 1024**3

    # List all HDF5 files in the folder
    h5_files = sorted(
        [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.endswith(".h5")]
    )
    random.shuffle(h5_files)
    print(h5_files[0:10])
    
    if not h5_files:
        print("No HDF5 files found in directory.")
        return

    # Open the first file to get dataset structure
    print(h5_files[0])
    with h5py.File(h5_files[0], "r") as first_file:

        # features
        # input
        #input_phase_shape = first_file["input_features_phase"].shape[1:]
        #input_spectrogram_shape = first_file["input_features_spectrogram"].shape[1:]
        #input_edges_shape = first_file["input_features_edges"].shape[1:]
        #input_cepstrum_shape = first_file["input_features_cepstrum"].shape[1:]
        #input_cepstrum_edges_shape = first_file["input_features_cepstrum_edges"].shape[1:]

        # output
        target_spectrogram_shape = first_file["target_features_spectrogram"].shape[1:]
        #noise_spectrogram_shape = first_file["noise_features_spectrogram"].shape[1:]

        # metadata
        filename_shape = first_file["filenames"].shape[1:]
        snr_db_shape = first_file["snr_db"].shape[1:]

        # dtypes
        # input
        #input_phase_dtype = first_file["input_features_phase"].dtype
        #input_spectrogram_dtype = first_file["input_features_spectrogram"].dtype
        #input_edges_dtype = first_file["input_features_edges"].dtype
        #input_cepstrum_dtype = first_file["input_features_cepstrum"].dtype
        #input_cepstrum_edges_dtype = first_file["input_features_cepstrum_edges"].dtype

        # output
        target_spectrogram_dtype = first_file["target_features_spectrogram"].dtype
        #noise_spectrogram_dtype = first_file["noise_features_spectrogram"].dtype

        # metadata
        filename_dtype = first_file["filenames"].dtype
        snr_db_dtype = first_file["snr_db"].dtype

        sample_size_bytes = (
            #np.prod(input_phase_shape) * np.dtype(input_phase_dtype).itemsize +
            #np.prod(input_spectrogram_shape) * np.dtype(input_spectrogram_dtype).itemsize +
            #np.prod(input_edges_shape) * np.dtype(input_edges_dtype).itemsize +
            #np.prod(input_cepstrum_shape) * np.dtype(input_cepstrum_dtype).itemsize +
            #np.prod(input_cepstrum_edges_shape) * np.dtype(input_cepstrum_edges_dtype).itemsize +
            np.prod(target_spectrogram_shape) * np.dtype(target_spectrogram_dtype).itemsize +
            #np.prod(noise_spectrogram_shape) * np.dtype(noise_spectrogram_dtype).itemsize +
            np.prod(filename_shape) * np.dtype(filename_dtype).itemsize +
            np.prod(snr_db_shape) * np.dtype(snr_db_dtype).itemsize
        )

    # Ensure output directory exists
    os.makedirs(output_folder_path, exist_ok=True)
    print('Made first file')

    # Variables for file management
    current_file_index = 0
    current_file_samples = 0
    current_file_size = 0
    previous_size = 0
    combined_file = None

    # Declare datasets at module scope
    filename_dataset = snr_db_dataset = None
    noise_spectrogram_dataset = input_phase_dataset = input_spectrogram_dataset = input_edges_dataset = input_cepstrum_dataset = input_cepstrum_edges_dataset = target_spectrogram_dataset = None

    def create_new_file():
        """Creates a new HDF5 output file."""
        nonlocal current_file_index, current_file_samples, current_file_size, combined_file
        nonlocal filename_dataset, snr_db_dataset, previous_size  # FIXED
        nonlocal input_phase_dataset, input_spectrogram_dataset, input_edges_dataset, input_cepstrum_dataset, input_cepstrum_edges_dataset, target_spectrogram_dataset, noise_spectrogram_dataset

        # Close previous file if it exists
        if combined_file is not None:
            combined_file.close()

        # Generate new file name
        file_path = os.path.join(output_folder_path, f"combined_{current_file_index:03d}.h5")
        combined_file = h5py.File(file_path, "w")

        # Input datasets
        #input_phase_dataset = combined_file.create_dataset(
        #    "input_features_phase", shape=(0, *input_phase_shape), chunks=(chunk_size, *input_phase_shape),
        #    maxshape=(None, *input_phase_shape), dtype=input_phase_dtype
        #)

        #input_spectrogram_dataset = combined_file.create_dataset(
        #    "input_features_spectrogram", shape=(0, *input_spectrogram_shape), chunks=(chunk_size, *input_spectrogram_shape),
        #    maxshape=(None, *input_spectrogram_shape), dtype=input_spectrogram_dtype
        #)

        #input_edges_dataset = combined_file.create_dataset(
        #    "input_features_edges", shape=(0, *input_edges_shape), chunks=(chunk_size, *input_edges_shape),
        #    maxshape=(None, *input_edges_shape), dtype=input_edges_dtype
        #)

        #input_cepstrum_dataset = combined_file.create_dataset(
        #    "input_features_cepstrum", shape=(0, *input_cepstrum_shape), chunks=(chunk_size, *input_cepstrum_shape),
        #    maxshape=(None, *input_cepstrum_shape), dtype=input_cepstrum_dtype
        #)

        #input_cepstrum_edges_dataset = combined_file.create_dataset(
        #    "input_features_cepstrum_edges", shape=(0, *input_cepstrum_edges_shape), chunks=(chunk_size, *input_cepstrum_edges_shape),
        #    maxshape=(None, *input_cepstrum_edges_shape), dtype=input_cepstrum_edges_dtype
        #)

        # Output dataset
        target_spectrogram_dataset = combined_file.create_dataset(
            "target_features_spectrogram", shape=(0, *target_spectrogram_shape), chunks=(chunk_size, *target_spectrogram_shape),
            maxshape=(None, *target_spectrogram_shape), dtype=target_spectrogram_dtype
        )
        #noise_spectrogram_dataset = combined_file.create_dataset(
        #    "noise_features_spectrogram", shape=(0, *noise_spectrogram_shape), chunks=(chunk_size, *noise_spectrogram_shape),
        #    maxshape=(None, *noise_spectrogram_shape), dtype=noise_spectrogram_dtype
        #)

        # Metadata datasets
        filename_dataset = combined_file.create_dataset(
            "filenames", shape=(0, *filename_shape), chunks=(chunk_size, *filename_shape),
            maxshape=(None, *filename_shape), dtype=filename_dtype
        )

        snr_db_dataset = combined_file.create_dataset(
            "snr_db", shape=(0, *snr_db_shape), chunks=(chunk_size, *snr_db_shape),
            maxshape=(None, *snr_db_shape), dtype=snr_db_dtype
        )

        current_file_samples = 0
        current_file_size = 0
        previous_size = 0
        current_file_index += 1

        print(f"Created new file: {file_path}")

    # Start processing files
    create_new_file()

    break_trigger = False
    for h5_file in h5_files:
        print(h5_file)
        with h5py.File(h5_file, "r") as source_file:
            # input
            #input_phase = source_file["input_features_phase"][:]
            #input_spectrogram = source_file["input_features_spectrogram"][:]
            #input_edges = source_file["input_features_edges"][:]
            #input_cepstrum = source_file["input_features_cepstrum"][:]
            #input_cepstrum_edges = source_file["input_features_cepstrum_edges"][:]

            # output
            target_spectrogram = source_file["target_features_spectrogram"][:]
            #noise_spectrogram = source_file["noise_features_spectrogram"]

            # metadata
            filename = source_file["filenames"][:]
            snr_db = source_file["snr_db"][:]

            # number samples
            num_samples = target_spectrogram.shape[0]

            for i in range(0, num_samples, chunk_size):
                # input
                #chunk_input_phase = input_phase[i:i+chunk_size]
                #chunk_input_spectrogram = input_spectrogram[i:i+chunk_size]
                #chunk_input_edges = input_edges[i:i+chunk_size]
                #chunk_input_cepstrum = input_cepstrum[i:i+chunk_size]
                #chunk_input_cepstrum_edges = input_cepstrum_edges[i:i+chunk_size]

                # output
                chunk_target_spectrogram = target_spectrogram[i:i+chunk_size]
                #chunk_noise_spectrogram = noise_spectrogram[i:i+chunk_size]

                # metadata
                chunk_filename = filename[i:i+chunk_size]
                chunk_snr_db = snr_db[i:i+chunk_size]

                # sample size
                chunk_sample_size = chunk_target_spectrogram.shape[0] * sample_size_bytes

                # Check if new file is needed
                if current_file_size + chunk_sample_size > max_file_size_bytes:
                    break_trigger = True
                    break
                    create_new_file()

                # Resize datasets| hi u farted
                new_size = current_file_samples + chunk_target_spectrogram.shape[0]

                # input
                #input_phase_dataset.resize((new_size, *input_phase_shape))
                #input_spectrogram_dataset.resize((new_size, *input_spectrogram_shape))
                #input_edges_dataset.resize((new_size, *input_edges_shape))
                #input_cepstrum_dataset.resize((new_size, *input_cepstrum_shape)) 
                #input_cepstrum_edges_dataset.resize((new_size, *input_cepstrum_edges_shape)) 

                # target
                target_spectrogram_dataset.resize((new_size, *target_spectrogram_shape))
                #noise_spectrogram_dataset.resize((new_size, *noise_spectrogram_shape))

                # metadata
                filename_dataset.resize((new_size, *filename_shape))
                snr_db_dataset.resize((new_size, *snr_db_shape))

                # Append data
                # input
                #input_phase_dataset[current_file_samples:new_size] = chunk_input_phase
                #input_spectrogram_dataset[current_file_samples:new_size] = chunk_input_spectrogram
                #input_edges_dataset[current_file_samples:new_size] = chunk_input_edges
                #input_cepstrum_dataset[current_file_samples:new_size] = chunk_input_cepstrum
                #input_cepstrum_edges_dataset[current_file_samples:new_size] = chunk_input_cepstrum_edges

                # target
                target_spectrogram_dataset[current_file_samples:new_size] = chunk_target_spectrogram
                #noise_spectrogram_dataset[current_file_samples:new_size] = chunk_noise_spectrogram

                # metadata
                filename_dataset[current_file_samples:new_size] = chunk_filename
                snr_db_dataset[current_file_samples:new_size] = chunk_snr_db

                current_file_samples = new_size
                current_file_size += chunk_sample_size

                # Print progress every ~1GB
                current_size_gb = current_file_size / 1024**3
                if math.floor(previous_size) != math.floor(current_size_gb):
                    print(f"Progress: {np.round(current_size_gb, 2)} GB - Processing {h5_file}")
                previous_size = current_size_gb
        if break_trigger:
            break
    # Close the last output file
    if combined_file is not None:
        combined_file.close()

    print(f"Finished combining files into {current_file_index} output files in {output_folder_path}")


import random
def combine_h5_files(h5_folder_path, output_folder_path, max_file_size_gb=1, chunk_size=128):
    """Combines multiple HDF5 files into a few large ones, ensuring they do not exceed max_file_size_gb."""
    
    # Convert max file size to bytes
    max_file_size_bytes = max_file_size_gb * 1024**3

    # List all HDF5 files in the folder
    h5_files = sorted(
        [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.endswith(".h5")]
    )
    np.random.shuffle(np.array(h5_files))
    print(h5_files[0:10])
    
    if not h5_files:
        print("No HDF5 files found in directory.")
        return

    # Open the first file to get dataset structure
    with h5py.File(h5_files[0], "r") as first_file:
        input_shape = first_file["input_images"].shape[1:]  # Exclude batch dim
        target_shape = first_file["target_images"].shape[1:]
        filename_shape = first_file["filenames"].shape[1:]
        snr_db_shape = first_file["snr_db"].shape[1:]

        input_dtype = first_file["input_images"].dtype
        target_dtype = first_file["target_images"].dtype
        filename_dtype = first_file["filenames"].dtype
        snr_db_dtype = first_file["snr_db"].dtype

        sample_size_bytes = (
            np.prod(input_shape) * np.dtype(input_dtype).itemsize +
            np.prod(target_shape) * np.dtype(target_dtype).itemsize +
            np.prod(filename_shape) * np.dtype(filename_dtype).itemsize +
            np.prod(snr_db_shape) * np.dtype(snr_db_dtype).itemsize
        )

    # Ensure output directory exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Variables for file management
    current_file_index = 0
    current_file_samples = 0
    current_file_size = 0
    previous_size = 0
    combined_file = None

    # Declare datasets at module scope
    input_dataset = target_dataset = filename_dataset = snr_db_dataset = None

    def create_new_file():
        """Creates a new HDF5 output file."""
        nonlocal current_file_index, current_file_samples, current_file_size, combined_file
        nonlocal input_dataset, target_dataset, filename_dataset, snr_db_dataset, previous_size  # FIXED

        # Close previous file if it exists
        if combined_file is not None:
            combined_file.close()

        # Generate new file name
        file_path = os.path.join(output_folder_path, f"combined_{current_file_index:03d}.h5")
        combined_file = h5py.File(file_path, "w")

        input_dataset = combined_file.create_dataset(
            "input_images", shape=(0, *input_shape), chunks=(chunk_size, *input_shape),
            maxshape=(None, *input_shape), dtype=input_dtype
        )
        target_dataset = combined_file.create_dataset(
            "target_images", shape=(0, *target_shape), chunks=(chunk_size, *target_shape),
            maxshape=(None, *target_shape), dtype=target_dtype
        )
        filename_dataset = combined_file.create_dataset(
            "filenames", shape=(0, *filename_shape), chunks=(chunk_size, *filename_shape),
            maxshape=(None, *filename_shape), dtype=filename_dtype
        )
        snr_db_dataset = combined_file.create_dataset(
            "snr_db", shape=(0, *snr_db_shape), chunks=(chunk_size, *snr_db_shape),
            maxshape=(None, *snr_db_shape), dtype=snr_db_dtype
        )

        current_file_samples = 0
        current_file_size = 0
        previous_size = 0
        current_file_index += 1

        print(f"Created new file: {file_path}")

    # Start processing files
    create_new_file()

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as source_file:
            input_data = source_file["input_images"][:]
            target_data = source_file["target_images"][:]
            filename_data = source_file["filenames"][:]  # Fixed typo
            snr_db_data = source_file["snr_db"][:]
            num_samples = input_data.shape[0]

            for i in range(0, num_samples, chunk_size):
                chunk_input = input_data[i:i+chunk_size]
                chunk_target = target_data[i:i+chunk_size]
                chunk_filename = filename_data[i:i+chunk_size]
                chunk_snr_db = snr_db_data[i:i+chunk_size]
                chunk_sample_size = chunk_input.shape[0] * sample_size_bytes

                # Check if new file is needed
                if current_file_size + chunk_sample_size > max_file_size_bytes:
                    break
                    create_new_file()

                # Resize datasets
                new_size = current_file_samples + chunk_input.shape[0]
                input_dataset.resize((new_size, *input_shape))
                target_dataset.resize((new_size, *target_shape))
                filename_dataset.resize((new_size, *filename_shape))
                snr_db_dataset.resize((new_size, *snr_db_shape))

                # Append data
                input_dataset[current_file_samples:new_size] = chunk_input
                target_dataset[current_file_samples:new_size] = chunk_target
                filename_dataset[current_file_samples:new_size] = chunk_filename
                snr_db_dataset[current_file_samples:new_size] = chunk_snr_db

                current_file_samples = new_size
                current_file_size += chunk_sample_size

                # Print progress every ~1GB
                current_size_gb = current_file_size / 1024**3
                if math.floor(previous_size) != math.floor(current_size_gb):
                    print(f"Progress: {np.round(current_size_gb, 2)} GB - Processing {h5_file}")
                previous_size = current_size_gb

    # Close the last output file
    if combined_file is not None:
        combined_file.close()

    print(f"Finished combining files into {current_file_index} output files in {output_folder_path}")

# dataset processor class 
import os

class DatasetProcessor:
    def __init__(self, train_music_dir, train_noise_dir, test_music_dir, test_noise_dir,
                 output_dir, SNRdB=(0, 20), batch_size=500, checkpoint_file_size=50000,
                 random_noise_level=0.0005,
                 background_noise_level=0.4, process_pool=True, verbose=False,
                 audio_length=int(44100 * 2), process_train=True, process_test=True,
                 mix_only=False):
        
        self.train_music_dir = train_music_dir
        self.train_noise_dir = train_noise_dir
        self.test_music_dir = test_music_dir
        self.test_noise_dir = test_noise_dir
        self.output_dir = output_dir
        self.SNRdB = SNRdB
        self.batch_size = batch_size
        self.checkpoint_file_size = checkpoint_file_size
        self.random_noise_level = random_noise_level
        self.background_noise_level = background_noise_level
        self.process_pool = process_pool
        self.verbose = verbose
        self.audio_length = audio_length
        self.process_train = process_train
        self.process_test = process_test
        self.mix_only = mix_only

        print('Output Dir:', self.output_dir)
        
        # Define output and checkpoint files
        self.train_checkpoint_file = os.path.join(self.output_dir, f"SNRdB_{SNRdB[0]}-{SNRdB[1]}/train-SNRdB_{SNRdB[0]}-{SNRdB[1]}-checkpoint.txt")
        self.train_output_file = os.path.join(self.output_dir, f"SNRdB_{SNRdB[0]}-{SNRdB[1]}/train/train-SNRdB_{SNRdB[0]}-{SNRdB[1]}.h5")
        
        self.test_checkpoint_file = os.path.join(self.output_dir, f"SNRdB_{SNRdB[0]}-{SNRdB[1]}/test-SNRdB_{SNRdB[0]}-{SNRdB[1]}-checkpoint.txt")
        self.test_output_file = os.path.join(self.output_dir, f"SNRdB_{SNRdB[0]}-{SNRdB[1]}/test/test-SNRdB_{SNRdB[0]}-{SNRdB[1]}.h5")
        
        # Print the file paths
        print("Train Checkpoint File:", self.train_checkpoint_file)
        print("Train Output File:", self.train_output_file)
        print("Test Checkpoint File:", self.test_checkpoint_file)
        print("Test Output File:", self.test_output_file)
        
        # make sure directories exist
        ensure_directory_exists(self.train_output_file)
        ensure_directory_exists(self.test_output_file)
    
    def process(self):

        if self.process_train:
            print('Processing training dataset....')
            process_and_save_separation_dataset(
                self.train_music_dir, self.train_noise_dir, self.train_output_file,
                self.train_checkpoint_file, audio_length=self.audio_length,
                batch_size=self.batch_size, background_noise_level=self.background_noise_level,
                random_noise_level=self.random_noise_level, SNRdB=self.SNRdB,
                process_pool=self.process_pool, verbose=self.verbose,
                checkpoint_file_size=self.checkpoint_file_size, 
                mix_only=self.mix_only
            )

        if self.process_test:
            print('Processing testing dataset....')
            process_and_save_separation_dataset(
                self.test_music_dir, self.test_noise_dir, self.test_output_file,
                self.test_checkpoint_file, audio_length=self.audio_length,
                batch_size=self.batch_size, background_noise_level=self.background_noise_level,
                random_noise_level=self.random_noise_level, SNRdB=self.SNRdB,
                process_pool=self.process_pool, verbose=self.verbose,
                checkpoint_file_size=self.checkpoint_file_size,
                mix_only=self.mix_only
            )

# Example usage
if __name__ == "__main__":
    processor = DatasetProcessor(
        train_music_dir='/content/drive/MyDrive/Datasets/Music/MUSDB18/train-2s-44100',
        train_noise_dir='/content/drive/MyDrive/Datasets/Noise/All_Noise/splits/train-2s-44100',
        test_music_dir='/content/drive/MyDrive/Datasets/Music/MUSDB18/test-2s-44100',
        test_noise_dir='/content/drive/MyDrive/Datasets/Noise/All_Noise/splits/test-2s-44100',
        output_dir='/content/drive/MyDrive/Datasets/Music-Noise/SNRdB_sep/',
        SNRdB=[0, 20]
    )
    processor.process()
