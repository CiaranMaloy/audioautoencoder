# Music Denoising Autoencoders: Models and Feature Extraction Methods

---

## Introduction
Source Separation has been a forefront of music AI research for the last 10 years. 
What makes music denoising a tricky problem is that it is difficult to define exactly what is music and what is noise, since there is a large amount of overlap between the two. E.g, a filter sweep in electronic music is normally based on uniform random noise, within context, this is music, or maybe some birds tweeting in the background of recodring an acoustic guitarist playing, although pleasant, within context, this would be noise. 
This is presented within the same space as source separation as the solution to the problem is very similar, and hence similar models are used. 
What is presented here is a comparison of several solutions to one problem within the source separation space, Music denoising. 

---

## Background and Literature Review
In note format, tidied up using ChatGPT. 

# 🎧 Audio Denoising & Source Separation Research Notes

This project explores **audio denoising framed as a source separation task**, drawing from current state-of-the-art models and research across both waveform and spectrogram domains.

---

## 📚 Literature Review

### 🔹 Part 1 – Foundational Papers and Takeaways

| Ref | Paper | Notes |
|-----|-------|-------|
| [a1] | [1807.01898 - Source Separation with Autoencoders](https://arxiv.org/abs/1807.01898) | Classic source separation via autoencoders. Consider DSD100 as an additional dataset. |
| [a2] | [2011.07274v2 - Sub-pixel Upsampling Improves Audio Quality](https://arxiv.org/abs/2011.07274) | Sub-pixel shufflers outperform transpose convolutions in upsampling stages. |
| [a3] | [BirdSoundsDenoising](https://openaccess.thecvf.com/content/WACV2023/papers/Zhang_BirdSoundsDenoising_Deep_Visual_Audio_Denoising_for_Bird_Sounds_WACV_2023_paper.pdf) | Denoising using segmentation-inspired architecture. Good reference for applying visual models to audio. |
| [a4] | [1806.04096 - Resynthesis with Variational Autoencoders](https://arxiv.org/abs/1806.04096) | VAEs are suitable for audio resynthesis. Suggests using input phase instead of predicting it. |
| [a5] | [Attention is All You Need](https://arxiv.org/abs/1706.03762) | Foundational transformer paper. Essential read if implementing attention in skip connections. |
| [a6] | *7-Ward-wimp2018.pdf* | Overview of SISEC MUSDB18 — the main benchmark dataset for source separation. |
| [a7] | [1805.02410 - MMDenseLSTM for Source Separation](https://arxiv.org/abs/1805.02410) | Enhancement of MMDenseNet with LSTM modules. |
| [a8] | [MMDenseNet Implementation (GitHub)](https://github.com/SeongwoongCho/MMDenseNet/blob/master/model.py) | Codebase for MMDenseNet, useful for reference and modification. |

#### 📌 Key Takeaways

- **Treat audio denoising as source separation**.
- **Use input phase** during reconstruction; most papers ignore phase prediction.
- **Equal loss weighting for phase and magnitude may hurt performance** — prioritize magnitude.
- **VarUNet** is viable, but get a stable deep version working first.
- **Attention in skip connections** is worth exploring (not common in current literature).
- Skip connections used so far: **convolutional** or **GRU-based**.
- Outputting the **noise estimate** can help guide learning and evaluation.

---

### 🔹 Part 2 – State-of-the-Art (Demucs and Hybrid Models)

| Ref | Paper | Notes |
|-----|-------|-------|
| [b1] | [2211.08553 - Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553) | Hybrid architecture mixing spectrograms and transformers. |
| [b2] | *Hybrid Spectrogram and Waveform Source Separation* | Emphasizes benefits of combining time and frequency domains. |
| [b3] | *Music Source Separation in the Waveform Domain* | Argues for end-to-end waveform separation for temporal fidelity. |
| [b4] | *Conv-TasNet (IEEE Xplore)* | A strong waveform-domain baseline for speech separation. |
| [b5] | [Open-Unmix - Reference for Data Augmentation](https://github.com/sigsep/open-unmix) | Practical techniques and code for augmenting MUSDB18. |

#### 🧠 Additional Thoughts

- **MUSDB18** is the gold standard for benchmarking. Augment it rather than adding unrelated music data.
- **DSD100** can supplement but doesn’t replace MUSDB18.
- Consider **Emotional Crowd Sound (IEEE DataPort)** for crowd noise-specific training or fine-tuning.

---

## 📂 Datasets

| Dataset | Use |
|---------|-----|
| [MUSDB18](https://sigsep.github.io/datasets/musdb.html) | Benchmark for music source separation |
| [DSD100](https://sigsep.github.io/datasets/dsd100.html) | Supplementary music dataset |
| [Emotional Crowd Sound (IEEE DataPort)](https://ieee-dataport.org/documents/emotional-crowd-sound-dataset) | Use for crowd noise denoising/fine-tuning |
| [ESC50](https://github.com/karolpiczak/ESC-50) | Dataset for Environmental Sound Classification

---

## ✅ Next Steps

- [ ] Implement VarUNet with magnitude-only loss and input phase.
- [ ] Explore skip connections with attention mechanisms (e.g., SE blocks or transformer attention).
- [ ] Begin with Segmentation model such as ref a3 for simplicity and to prevent against perceptual artefacts reported in all papers regarding direct re-synthesis. As I only want to get rid of the noise, and not separate and re-synthesise it. 

---

---

## Methodology

### Datasets 
- Datasets Used: 
    - MUSDB18
    - ESC-50
    - Some Custom Noise and Music samples

Dataset Synthesis and/or labeling is primarily the most important part of any supervised learning task. There are no large, commonly used datasets for music denoising which come pre-noised and so multiple datasets are used in this study as music and noise and then mixed together to augment each dataset. 

Datasets are separated into training and testing instances from the get go with a split of 80:20. MUSDB18 comes preallocated as training and testing, all other datasets are split randomly on a whole sample basis (before any preprocessing)

After data has been separated into folders of music training, music testing, noise training and noise testing, training and testing data is synthesised on a basis of a signal to noise ratio (SNR) range, in this case -10 dB to 10 dB of SNR is selected. 

Music instances and noise instances are then mixed such that each mixing has a signal to noise ratio chosen at random within a pre-determined range. After mixing, features are extracted from training data as input (noisy) and target (clean) along with metadata and saved as .h5 files in google drive. 

### Features and metadata
One novel aspect of this research is the use of perceptual band filtering as input channels. 
Each spectrogram is split into 4 bands as: 
1. Full signal 
2. 5kHz and below
3. 1.25kHz and below
4. 500 Hz and below

All resampled to have the same height and with as images. This is to encourage the loss function to perceptually weight the importance of each frequency band. This also introduces oversampling in the lower frequency bands and undersampling within higher frequency bands, where the perceptual difference between two frequencies is less, even if it is weighted equally in the fourier transform. 

### Dataset Suitability
The long and short of how suitable the dataset is for training an autoencoder for denoising, is that it isn't inadvertently learning any sort of relationship that may simplify the task or encourage any non-ideal behaviour. 
Below is a table of extracted features, and their correlation with the signal to noise ratio (SNR) with the intention of validating the suitability of the dataset as a training and testing dataset. 

| Feature       | Correlation |
|---------------|-------------|
| in_entropy    |  0.074   |
| in_energy     | -0.064  |
| in_sparsity   | -0.062  |
| in_kurtosis   | -0.051 |
| in_var        | -0.038   |
| in_skew       |  0.032   |
| in_mean       | -0.031   |
| in_median     | -0.027   |
| in_std        | -0.023   |
| instance      | -0.005   |
| in_iqr        | -0.001   |
| in_range      | -0.001   |
| in_max        | -0.001   |
| in_min        | NaN         |
| tar_min       | NaN         |


### Models and Training
- Model architectures used
- Training setup and hyperparameters
- Evaluation criteria and metrics

#### UNetConv4

Total params: 1,815,004   
Trainable params: 1,815,004   
Non-trainable params: 0   
Total mult-adds (Units.GIGABYTES): 2.52   

Input size (MB): 1.43   
Forward/backward pass size (MB): 25.21   
Params size (MB): 7.26   
Estimated Total Size (MB): 33.91   

## Results
Present and compare your model results. Consider structuring this into two subsections:

### Model Comparison

Models are trained on 100 GB of data and tested on 50 GB of data. 
L1 loss is being used as a metric.
| Model Name | Train Loss | Val Loss | Test Loss |
|------------|----------|----------|----------|
| UNetConv4 (multi-channel)  | 0.134    | 0.139    |  0.158   |
| UNetRes12 (multi-channel) | 0.146    | 0.148    |  0.172   |
| UNetConv4 (resampled to mel-scale)  |  0.122    |  0.117  |  0.152  |
| Model D   (resampled to mel-scale)|          |          |          |

Models are shown as multi-channel (with different bands sampled as different channels) and mel-scale as one image resampled to be mel scale to be used with a 

### Feature Combination Comparison (Best Model)

| Feature Combination | Metric 1 | Metric 2 | Metric 3 |
|---------------------|----------|----------|----------|
| Spectrogram only    |          |          |          |
| Cepstrum            |          |          |          |
| Edge Detection      |          |          |          |

Include plots or visualizations as appropriate.

---

## Conclusion and Discussion
Summarize key findings, discuss their implications, and highlight potential areas for future research or improvement.

---

## References
Provide a list of references used in your literature review and throughout your methodology.
