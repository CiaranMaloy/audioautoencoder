import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def process_and_reconstruct_audio(
    denoised_imgs, noisy_imgs, clean_imgs, image_to_waveform, sr=44100, i=0, cmap='twilight'
):
    """
    Processes spectrogram images and reconstructs audio for visualization and comparison.

    Parameters:
    - denoised_imgs: Array of denoised spectrogram images [shape: (N, 2, H, W)].
    - noisy_imgs: Array of noisy spectrogram images [shape: (N, 2, H, W)].
    - clean_imgs: Array of clean spectrogram images [shape: (N, 2, H, W)].
    - image_to_waveform: Function to convert spectrogram images to waveform audio.
    - sr: Sample rate for spectrograms (default: 44100).
    - i: Index of the image to process (default: 0).
    - cmap: Colormap for visualization (default: 'twilight').

    Returns:
    - clean_audio: Reconstructed clean audio waveform.
    - noisy_audio: Reconstructed noisy audio waveform.
    - new_audio: Reconstructed denoised audio waveform.
    """
    # Extract images
    image = denoised_imgs[i]
    image_noisy = noisy_imgs[i]
    image_clean = clean_imgs[i]

    print("Max value in denoised image:", np.max(image))
    print("Min value in denoised image:", np.min(image))

    # Plotting spectrograms
    def plot_spectrograms(images, title_prefix):
        plt.figure(figsize=(12, 8))
        for idx, img in enumerate(images):
            plt.subplot(3, 1, idx + 1)
            librosa.display.specshow(img, sr=sr, y_axis='log', x_axis='time', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'{title_prefix} - {["Magnitude", "Normalized", "Phase"][idx]} Spectrogram')
        plt.tight_layout()
        plt.show()

    plot_spectrograms(image, "Denoised")
    plot_spectrograms(image_noisy, "Noisy")
    plot_spectrograms(image_clean, "Clean")

    # Plot mean values along the frequency axis
    def plot_mean_along_frequency_axis(image_data, label):
        plt.plot(np.mean(image_data[0], axis=1), label=f'{label} - Magnitude')
        plt.plot(np.mean(image_data[1], axis=1), label=f'{label} - Normalized')
        plt.legend()
        plt.show()

    plot_mean_along_frequency_axis(image_noisy, "Noisy")
    plot_mean_along_frequency_axis(image_clean, "Clean")

    # Reconstruct audio from images
    new_audio = image_to_waveform(image, sr)
    new_audio = new_audio / np.max(np.abs(new_audio))
    noisy_audio = image_to_waveform(image_noisy, sr)
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    clean_audio = image_to_waveform(image_clean, sr)
    clean_audio = clean_audio / np.max(np.abs(clean_audio))

    print("Audio processed!")

    # Plot waveforms
    n = 0
    c = len(clean_audio)
    def plot_waveform(audio, title):
        plt.plot(audio)
        plt.xlim((n, n + c))
        plt.title(title)
        plt.show()

    plot_waveform(clean_audio, "Clean Audio")
    plot_waveform(new_audio, "Denoised Audio")
    plot_waveform(noisy_audio, "Noisy Audio")

    return clean_audio, noisy_audio, new_audio