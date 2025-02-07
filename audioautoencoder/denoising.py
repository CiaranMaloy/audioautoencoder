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
    def __init__(self, model, output_path, sample_rate=44100, chunk_duration=2, step_size=0.5, device=None):
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
            input_image = audio_to_image(chunk, sr, self.chunk_samples, features=False)
            
            # AI denoising
            input_tensor = torch.tensor(np.array([input_image]), dtype=torch.float32).to(self.device)
            denoised_img = self.model(input_tensor)
            
            input_image = input_tensor.cpu().numpy()[0]
            denoised_img = denoised_img.cpu().numpy()[0]

            # Convert back to waveform
            output_chunk = magphase_to_waveform(denoised_img[0], input_image[1], self.chunk_samples)
            output_chunk_input = magphase_to_waveform(input_image[0], input_image[1], self.chunk_samples)

            processed_input.append(output_chunk_input)
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
