# Improving Music Denoising Autoencoders: A Comparative Study of Models and Feature Extraction Methods

---

## Introduction
Source Separation has been a forefront of music AI research for the last 10 years. 
What makes music denoising a tricky problem is that it is difficult to define exactly what is music and what is noise, since there is a large amount of overlap between the two. E.g, a filter sweep in electronic music is normally based on uniform random noise, within context, this is music, or maybe some birds tweeting in the background of recodring an acoustic guitarist playing, although pleasant, within context, this would be noise. 
This is presented within the same space as source separation as the solution to the problem is very similar, and hence similar models are used. 
What is presented here is a comparison of several solutions to one problem within the source separation space, Music denoising. 

---

## Background and Literature Review
Summarize existing research and methods relevant to your work. Highlight gaps in current research that your project intends to fill.

- Paper/Reference 1
- Paper/Reference 2
- Paper/Reference 3

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

Total params: 1,815,004 //
Trainable params: 1,815,004 //
Non-trainable params: 0 //
Total mult-adds (Units.GIGABYTES): 2.52 //

Input size (MB): 1.43 //
Forward/backward pass size (MB): 25.21 //
Params size (MB): 7.26 //
Estimated Total Size (MB): 33.91 //

## Results
Present and compare your model results. Consider structuring this into two subsections:

### Model Comparison

Models are trained on 100 GB of data and tested on 50 GB of data. 
L1 loss is being used as a metric.
| Model Name | Train Loss | Val Loss | Test Loss |
|------------|----------|----------|----------|
| UNetConv4  | 0.134    | 0.139    |  0.158   |
| Model B    |          |          |          |
| Model C    |          |          |          |
| Model D    |          |          |          |

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
