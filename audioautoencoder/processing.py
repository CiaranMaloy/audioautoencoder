import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
import random

# Parameters
def generate_input_audio():
  # Parameters
  t = 1  # Duration in seconds
  sr = 44100  # Sampling rate
  T = np.linspace(0, t, int(sr * t), endpoint=False)

  # Randomize frequencies
  f = random.uniform(100, 2000)  # Random fundamental frequency
  f2 = random.uniform(50, 500)   # Random secondary frequency
  f3 = random.uniform(10000, 20000)  # Random secondary frequency

  # Random amplitude modulation for each frequency component
  amp_f1 = np.sin(2 * np.pi * random.uniform(0.5, 1.5) * T) # Amplitude modulation factor for f
  amp_f2 = np.sin(2 * np.pi * random.uniform(0.5, 2) * T) # Amplitude modulation factor for f
  amp_f4 = np.sin(2 * np.pi * random.uniform(0.5, 1.5) * T) # Amplitude modulation factor for f
  amp_f2_secondary = np.sin(2 * np.pi * random.uniform(0.1, 0.5) * T) # Amplitude modulation factor for f
  amp_f2_3 = np.sin(2 * np.pi * random.uniform(0.5, 4) * T) # Amplitude modulation factor for f
  amp_f3_3 = np.sin(2 * np.pi * random.uniform(0.5, 4) * T) # Amplitude modulation factor for f

  # Generate audio signal with amplitude modulation
  audio = (amp_f1 * np.sin(2 * np.pi * f * T) +
          amp_f2 * np.sin(2 * np.pi * 2 * f * T) +
          amp_f4 * np.sin(2 * np.pi * 4 * f * T) +
          amp_f2_secondary * np.sin(2 * np.pi * f2 * T) +
          amp_f2_3 * np.sin(2 * np.pi * 10 * f2 * T) * 0.5 +
          amp_f3_3 * np.sin(2 * np.pi * f3 * T) * 0.2)

  # Normalize audio to [-1, 1]
  audio = audio / np.max(np.abs(audio))
  return audio, sr

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


def audio_to_image(audio, sr, verbose=False, n_fft=2048, audio_length=44100, features=True):
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
    n = len(audio)
    assert(n == audio_length)
    audio_pad = librosa.util.fix_length(audio, size=n + n_fft // 2)
    stft = librosa.stft(audio_pad, n_fft=n_fft)
    magnitude, phase = np.abs(stft), np.angle(stft) 
    logmagnitude = 10 * np.log10(magnitude + 1e-8)

    #print('stft size:', np.shape(stft))

    #print(np.shape(logmagnitude))

    if verbose:
        print('Log magnitude max and min:', np.max(logmagnitude), np.min(logmagnitude))

    # Normalise magnitude by frequency
    freqs = np.linspace(0, sr // 2, 1025)
    weights = freqs.copy()
    weights[weights == 0] = 1  # Avoid division by zero for DC component
    normalised_magnitude = (magnitude.T * weights).T
    normalised_magnitude = 10 * np.log10(normalised_magnitude + 1e-8)

    # Clip and normalise ranges
    logmagnitude = np.clip((logmagnitude + 30) / 60, 0, 1)
    phase = np.clip((phase + np.pi) / (2 * np.pi), 0, 1)

    # Stack as 3 channels: log magnitude, normalised magnitude, and phase
    if features:
        normalised_magnitude = np.clip((normalised_magnitude - 20) / 20, 0, 1)
        output = np.stack([logmagnitude, normalised_magnitude, phase], axis=0)
    else:
        output = np.stack([logmagnitude, phase], axis=0)
    assert(np.shape(output) == (2, 1025, 175))
    return output

def process_audio_to_image(audio, sr, plot=False, noise_level=0, audio_length=44100):
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
    input_image = audio_to_image(noisy_audio, sr, audio_length=audio_length)
    target_image = audio_to_image(audio, sr, audio_length=audio_length)

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

def image_to_waveform(image, audio_length=44100):
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
    return librosa.istft(stft, length=audio_length)
