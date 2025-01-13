import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
import random

def bandpass_filter(data, lowcut, highcut, sample_rate, order=1):
    """
    Applies a bandpass filter to the input signal.
    
    Parameters:
        data (np.array): Input audio data.
        lowcut (float): Low cut-off frequency in Hz.
        highcut (float): High cut-off frequency in Hz.
        sample_rate (float): Sampling rate of the signal.
        order (int): Order of the Butterworth filter.

    Returns:
        np.array: Filtered signal.
    """
    sos = butter(order, [lowcut, highcut], btype='band', fs=sample_rate, output='sos')
    return sosfilt(sos, data)

def audio_to_image(audio, sr, length=86, verbose=False):
    """
    Converts audio signal into a 3-channel image representation.

    Parameters:
        audio (np.array): Input audio signal.
        sr (int): Sampling rate of the signal.
        length (int): Desired length of the output spectrogram.
        verbose (bool): Whether to print debugging information.

    Returns:
        np.array: A 3D array (3, 1024, length) representing the spectrogram.
    """
    stft = librosa.stft(audio, n_fft=2048)
    magnitude, phase = np.abs(stft)[:1024, :length], np.angle(stft)[:1024, :length]
    logmagnitude = 10 * np.log10(magnitude + 1e-8)

    if verbose:
        print('Log magnitude max and min:', np.max(logmagnitude), np.min(logmagnitude))

    # Normalise magnitude by frequency
    freqs = np.linspace(0, sr // 2, 1024)
    weights = freqs.copy()
    weights[weights == 0] = 1  # Avoid division by zero for DC component
    normalised_magnitude = (magnitude.T * weights).T
    normalised_magnitude = 10 * np.log10(normalised_magnitude + 1e-8)

    # Clip and normalise ranges
    logmagnitude = np.clip((logmagnitude + 30) / 60, 0, 1)
    normalised_magnitude = np.clip((normalised_magnitude - 20) / 20, 0, 1)
    phase = np.clip((phase + np.pi) / (2 * np.pi), 0, 1)

    # Stack as 3 channels: log magnitude, normalised magnitude, and phase
    return np.stack([logmagnitude, normalised_magnitude, phase], axis=0)

def process_audio_to_image(audio, sr, plot=False, noise_level=0):
    """
    Processes an audio signal into input and target image pairs.

    Parameters:
        audio (np.array): Input audio signal.
        sr (int): Sampling rate of the signal.
        plot (bool): Whether to display the resulting spectrograms.
        noise_level (float): Noise level to apply to the input audio.

    Returns:
        tuple: Input image and target image arrays.
    """
    T = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
    sigma = 1
    s = np.random.normal(0, sigma, len(audio)) * np.sin(2 * np.pi * random.uniform(0.3, 4) * T)
    s = (s / np.max(np.abs(s))) * noise_level if noise_level != 0 else s

    noisy_audio = np.clip(audio + s, -1, 1) if random.uniform(0, 1) > 0.2 else audio
    input_image = audio_to_image(noisy_audio, sr)
    target_image = audio_to_image(audio, sr)

    if plot:
        for i, (img, title) in enumerate(zip(
            [input_image[0], input_image[1], input_image[2], target_image[0], target_image[1], target_image[2]],
            ["Input Log Magnitude", "Input Normalised Magnitude", "Input Phase",
             "Target Log Magnitude", "Target Normalised Magnitude", "Target Phase"]
        )):
            plt.subplot(6, 1, i + 1)
            librosa.display.specshow(img, sr=sr, y_axis='log', x_axis='time', cmap='twilight')
            plt.colorbar()
            plt.title(title)
        plt.tight_layout()
        plt.show()

    return input_image, target_image

def denormalise(image):
    """
    Denormalises the spectrogram image back to original scale.

    Parameters:
        image (np.array): Normalised 3D image array.

    Returns:
        np.array: Denormalised image array.
    """
    image[0] = 10 ** ((image[0] * 60 - 30) / 10)
    image[2] = image[2] * (2 * np.pi) - np.pi
    return image

def image_to_waveform(image, sr):
    """
    Converts a spectrogram image back into an audio waveform.

    Parameters:
        image (np.array): Spectrogram image (3 channels).
        sr (int): Sampling rate.

    Returns:
        np.array: Reconstructed audio waveform.
    """
    image = denormalise(image)
    magnitude = image[0]
    phase = image[2]
    stft = magnitude * np.exp(1j * phase)
    return librosa.istft(stft)
