import numpy as np
import random
import matplotlib.pyplot as plt
from utils import *
from processing import *

def generate_audio_with_noise(audio_file, noise_file, start_time=10, duration=10, 
                             signal_level=1, noise_level=0.4, sr=44100, plot=False):
    """
    Loads an audio file and a noise file, trims them, normalizes, and adds Gaussian noise.
    
    Parameters:
        audio_file (str): Path to the main audio file.
        noise_file (str): Path to the noise file.
        start_time (int): Start time (in seconds) for trimming.
        duration (int): Total duration (in seconds).
        signal_level (float): Scaling factor for the audio signal.
        noise_level (float): Scaling factor for the noise.
        sr (int): Expected sample rate (default: 44100 Hz).

    Returns:
        noisy_audio (np.array): Processed noisy audio.
        snr (float): Signal-to-noise ratio in dB.
    """
    # Load audio and noise
    audio, audio_sr = load_audio_file(audio_file)
    noise_waveform, noise_sr = load_audio_file(noise_file)
    
    print('Noise Sample Rate:', noise_sr)
    
    assert audio_sr == sr, f"Expected sample rate {sr}, but got {audio_sr}"

    # Trim audio and noise to the specified start time and duration
    audio = audio[start_time * sr : (start_time + duration) * sr]
    noise_waveform = noise_waveform[start_time * noise_sr : (start_time + duration) * noise_sr]

    # Normalize audio to [-1, 1]
    audio = np.clip((audio / np.max(np.abs(audio))) * signal_level, -1, 1)
    noise_waveform = np.clip((noise_waveform / np.max(np.abs(noise_waveform))) * noise_level, -1, 1)

    # Generate Gaussian noise
    T = np.linspace(0, 1, num=len(audio))
    mu, sigma = 0, random.uniform(0, 0.8)
    gaussian_noise = np.random.normal(mu, sigma, len(audio)) * np.sin(2 * np.pi * random.uniform(0.3, 4) * T)
    gaussian_noise = np.nan_to_num(gaussian_noise)
    gaussian_noise = (gaussian_noise / np.max(np.abs(gaussian_noise))) * noise_level

    # Add noise to the signal
    noisy_audio = np.clip(audio + noise_waveform + gaussian_noise, -1, 1)

    # Compute SNR
    signal_power = np.mean(audio**2)
    noise_power = np.mean(noise_waveform**2)
    snr = 10 * np.log10(signal_power / noise_power)

    print(f"SNR: {snr:.2f} dB")

    # Plot results
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(noise_waveform, label="Noise")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(audio, label="Clean Audio")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(noisy_audio, label="Noisy Audio")
        plt.legend()
        plt.show()

    return noisy_audio, sr

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import soundfile as sf

class AudioDenoiser:
    def __init__(self, model, scalers, output_path, sample_rate=44100, chunk_duration=2, step_size=0.5, device=None):
        """
        Audio Denoising Pipeline using AI model.
        
        Parameters:
            model (torch.nn.Module): AI model for denoising.
            output_path (str): Directory to save output files.
            sample_rate (int): Sample rate (default 44100 Hz).
            chunk_duration (int): Duration of each chunk in seconds.
            step_size (float): Step size for overlap-add in seconds.
            device (str, optional): Device for PyTorch computation ("cuda" or "cpu").
        """
        self.model = model
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.chunk_samples = sample_rate * chunk_duration
        self.scalers = scalers
        self.step_samples = int(self.chunk_samples * step_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def process_audio(self, waveform, sr):
        """
        Processes audio by adding noise, chunking, denoising, and reconstructing.

        Parameters:
            input_path (str): Path to the input song file.
            noise_path (str): Path to the noise file.

        Returns:
            Tuple of (reconstructed_audio, reconstructed_audio_input).
        """
        # Load audio
        self.waveform = waveform
        assert sr == self.sample_rate, f"Sample rate mismatch: expected {self.sample_rate}, got {sr}"

        # Process in chunks
        processed_audio, processed_input = [], []
        for start in range(0, len(waveform) - self.chunk_samples + 1, self.step_samples):
            chunk = waveform[start:start + self.chunk_samples]
            features = extract_features(chunk, sr, audio_length=self.chunk_samples)
            transformed_features, metadata = transform_features(features)

            # AI denoising
            input_tensor = torch.tensor(np.array([transformed_features]), dtype=torch.float32).to(self.device)
            denoised = self.model(input_tensor)
            
            input_image = input_tensor.cpu().numpy()[0]
            denoised = denoised.cpu().numpy()[0]

            denoised_spectrogram = reconstruct_spectrogram(denoised, metadata, self.scalers)

            # Convert back to waveform
            output_chunk = magphase_to_waveform(denoised_spectrogram, features['phase'], self.chunk_samples)
            #output_chunk_input = magphase_to_waveform(input_image[0], features['phase'], self.chunk_samples)

            processed_input.append(chunk)
            processed_audio.append(output_chunk)

        # Reconstruct waveform with overlap-add
        reconstructed_audio = self._overlap_add(processed_audio)
        reconstructed_audio_input = self._overlap_add(processed_input)

        # Save output
        self._save_audio(reconstructed_audio, "output_audio_song.wav")
        self._save_audio(reconstructed_audio_input, "input_audio_song.wav")

        # Plot spectrograms
        self._plot_spectrograms(reconstructed_audio, reconstructed_audio_input)

        return reconstructed_audio, reconstructed_audio_input

    def _overlap_add(self, chunks):
        """Reconstructs the waveform using overlap-add method."""
        reconstructed = np.zeros(len(self.waveform))
        weight = np.zeros(len(self.waveform))

        for i, start in enumerate(range(0, len(self.waveform) - self.chunk_samples + 1, self.step_samples)):
            reconstructed[start:start + self.chunk_samples] += chunks[i]
            weight[start:start + self.chunk_samples] += np.hanning(self.chunk_samples)

        reconstructed /= np.maximum(weight, 1e-6)
        reconstructed = np.clip(reconstructed, -1, 1)
        
        fade_in = int(self.sample_rate / 2)
        reconstructed[:fade_in] *= np.hanning(self.sample_rate)[:fade_in]
        reconstructed[-fade_in:] *= np.hanning(self.sample_rate)[-fade_in:]

        return reconstructed

    def _save_audio(self, audio, filename):
        """Saves the audio file."""
        output_filename = os.path.join(self.output_path, add_datetime_to_filename(filename))
        sf.write(output_filename, audio / np.max(audio), self.sample_rate)
        print(f"Saved: {output_filename}")

    def _plot_spectrograms(self, reconstructed_audio, reconstructed_audio_input):
        """Plots spectrograms of processed and input audio."""
        plt.figure(figsize=(10, 6))
        plt.specgram(reconstructed_audio, Fs=self.sample_rate, NFFT=2048, noverlap=1024, cmap="viridis")
        plt.yscale("log")
        plt.ylim((20, 20000))
        plt.title("Spectrogram of Processed Audio")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Amplitude (dB)")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.specgram(reconstructed_audio_input, Fs=self.sample_rate, NFFT=2048, noverlap=1024, cmap="viridis")
        plt.yscale("log")
        plt.ylim((20, 20000))
        plt.title("Spectrogram of Input")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Amplitude (dB)")
        plt.show()

def resample_feature(feature, target_shape):
    """Resamples a 2D numpy feature array to match target shape using torch.nn.functional.interpolate."""
    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    target_size = (target_shape[0], target_shape[1])  # (new_H, new_W)
    
    resized_feature = F.interpolate(feature_tensor, size=target_size, mode="bilinear", align_corners=False)
    return resized_feature.squeeze(0).squeeze(0).numpy()  # Remove batch/channel dim and return as numpy

def transform_features_bandchannels(features, scalers):
    input_spectrogram = features['spectrogram']
    input_edges = features['edges']
    input_cepstrum = features['cepstrum']

    # function to transform the extracted features to an input to the 
    target_shape = input_spectrogram.shape
    # Apply scalers
        #input_phase = self.scalers["input_features_phase"].transform(input_phase.reshape(1, -1)).reshape(input_phase.shape)
    input_spectrogram = scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)
    #input_edges = scalers["input_features_edges"].transform(input_edges.reshape(1, -1)).reshape(input_edges.shape)
    #input_cepstrum = scalers["input_features_cepstrum"].transform(input_cepstrum.reshape(1, -1)).reshape(input_cepstrum.shape)
    #input_cepstrum_edges = self.scalers["input_features_cepstrum_edges"].transform(input_cepstrum_edges.reshape(1, -1)).reshape(input_cepstrum_edges.shape)

    # resample mfcc featues so theyre the same shape as the spectrogram and phase features
    # Define frequency bins
    sampling_rate = 44100  # 44.1 kHz audio
    n_fft = 2048  # Adjust this for better resolution
    freqs = np.linspace(0, sampling_rate / 2, n_fft // 2 + 1)  # STFT frequency bins

    # Find indices corresponding to 0–4000 Hz
    min_freq, hf, mf, lf = 0, 5000, 1250, 500 
    freq_indices_hf = np.where((freqs >= min_freq) & (freqs <= hf))[0]
    freq_indices_mf = np.where((freqs >= min_freq) & (freqs <= mf))[0]
    freq_indices_lf = np.where((freqs >= min_freq) & (freqs <= lf))[0]
    # input spectrogram
    input_spectrogram_hf = resample_feature(input_spectrogram[freq_indices_hf, :], target_shape)
    input_spectrogram_mf = resample_feature(input_spectrogram[freq_indices_mf, :], target_shape)
    input_spectrogram_lf = resample_feature(input_spectrogram[freq_indices_lf, :], target_shape)
    # edges
    #input_edges_hf = resample_feature(input_edges[freq_indices_hf, :], target_shape)
    #input_edges_mf = resample_feature(input_edges[freq_indices_mf, :], target_shape)
    #input_edges_lf = resample_feature(input_edges[freq_indices_lf, :], target_shape)

    # now input indices for 0-1000 and 0-200 to add as channels and as freq_indicies for reconstruction

    # Resample MFCC features
    #input_cepstrum = resample_feature(input_cepstrum, target_shape)
    
    # Convert to tensors - input_phase, is missing,..... it's too confusing
    inputs = torch.tensor(np.stack([
        input_spectrogram, input_spectrogram_hf, input_spectrogram_mf, input_spectrogram_lf,
    ], axis=0), dtype=torch.float32)  # Shape: (6, H, W)

    a = 2
    inputs = (inputs/a) + 0.5

    # metadata
    # Extract metadata
    metadata = {
        "hf_shape": input_spectrogram[freq_indices_hf, :].shape,
        "mf_shape": input_spectrogram[freq_indices_mf, :].shape,
        "lf_shape": input_spectrogram[freq_indices_lf, :].shape,
        "freq_indices_hf": freq_indices_hf,
        "freq_indices_mf": freq_indices_mf,
        "freq_indices_lf": freq_indices_lf
    }

    return inputs, metadata

from datasets.loaders import HDF5Dataset_mel_warp
def transform_features_mel_scale(features, scalers):
    input_spectrogram = features['spectrogram']

    # function to transform the extracted features to an input to the 
    input_spectrogram = scalers["input_features_spectrogram"].transform(input_spectrogram.reshape(1, -1)).reshape(input_spectrogram.shape)

    # Find indices corresponding to 0–4000 Hz
    input_spectrogram = HDF5Dataset_mel_warp.warp_spectrogram(input_spectrogram, sr=44100)
    
    # Convert to tensors 
    inputs = torch.tensor(np.stack([
        input_spectrogram
    ], axis=0), dtype=torch.float32)  # Shape: (6, H, W)

    a = 2
    inputs = (inputs/a) + 0.5

    # metadata
    # Extract metadata
    metadata = {
    }

    return inputs, metadata

def unwarp_mask(mask):
    return HDF5Dataset_mel_warp.unwarp_spectrogram(mask, sr=44100)

def reconstruct_spectrogram(outputs, metadata):
    # lets evaluate this from a l1 loss perspective
    # reconstruct spectrogram
    out_spectrogram = np.array(outputs[0])
    out_spectrogram[metadata["freq_indices_hf"], :] = resample_feature(outputs[1], metadata["hf_shape"])
    out_spectrogram[metadata["freq_indices_mf"], :] = resample_feature(outputs[2], metadata["mf_shape"])
    out_spectrogram[metadata["freq_indices_lf"], :] = resample_feature(outputs[3], metadata["lf_shape"])
    return out_spectrogram

def inverse_scale(out_spectrogram, scalers):
    # inverse scale the
    # transform back to 0 centred and
    out_spectrogram = (out_spectrogram - 0.5) * 2
    out_spec_shape = out_spectrogram.shape

    # undo scaler
    out_spectrogram = scalers["input_features_spectrogram"].inverse_transform(np.array([out_spectrogram]).reshape(1, -1)).reshape(out_spec_shape)
    return out_spectrogram
