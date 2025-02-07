# 1. create training, testing and validating datasets
import os
import glob
import random
import pickle

def create_datasets(dataset_dirs, output_dir, random_seed=42, resume_file="split_state.pkl"):
    """
    Collects wav files from dataset directories, removes duplicates, splits them into training, validation, and testing sets,
    and ensures reproducibility and resumption of testing.

    Args:
        dataset_dirs (list of str): List of directories containing wav files.
        output_dir (str): Directory to save split files and resume state.
        random_seed (int): Seed for reproducibility.
        resume_file (str): File to save/load resume state.

    Returns:
        dict: Dictionary with training, validation, and testing sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    resume_path = os.path.join(output_dir, resume_file)

    # If resume file exists, load previous state
    if os.path.exists(resume_path):
        with open(resume_path, "rb") as f:
            splits = pickle.load(f)
        print("Resuming from saved state.")
        return splits

    # Gather all wav files
    all_files = []
    for dataset_dir in dataset_dirs:
        all_files.extend(glob.glob(os.path.join(dataset_dir, "**" ,"*.wav"), recursive=True))

    if not all_files:
        raise ValueError("No .wav files found in the provided directories.")

    # Remove duplicates
    all_files = list(set(all_files))

    # Shuffle and split files
    random.seed(random_seed)
    random.shuffle(all_files)

    num_files = len(all_files)
    num_train = int(num_files * 0.7)
    num_val = int(num_files * 0.1)

    splits = {
        "train": all_files[:num_train],
        "val": all_files[num_train:num_train + num_val],
        "test": all_files[num_train + num_val:]
    }

    # Save splits for reproducibility
    split_file = os.path.join(output_dir, "splits.pkl")
    with open(split_file, "wb") as f:
        pickle.dump(splits, f)

    # Save resume state
    with open(resume_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"Splits saved to {split_file}. Resume state saved to {resume_path}.")
    return splits