# Yes / No Speech Classifier

A simple **speech classification project using a GRU-based Recurrent Neural Network (RNN)** to classify short audio recordings as either **"Yes"** or **"No"**.

The project demonstrates an end-to-end audio classification pipeline, starting from raw speech recordings and ending with binary classification using a GRU.

## Overview

The project follows these main steps:

1. Split recordings into individual words using **energy-based silence detection**
2. Convert each word into a **Mel-spectrogram**
3. Split the data into **training, validation, and test sets**
4. Normalize the Mel-spectrogram features using statistics computed from the training set
5. Train a **GRU-based RNN classifier**
6. Evaluate the classifier on the test set

The model achieved **100% test accuracy** on the test set used in this project.

## Model Architecture

The classifier uses a simple GRU-based recurrent neural network:

```text
Mel-spectrogram
      │
      ▼
  GRU Layer
  Input size: 40
  Hidden size: 32
      │
      ▼
Last time-step output
      │
      ▼
Linear Layer
Output size: 1
      │
      ▼
Binary Prediction
Yes / No
```

### Architecture Details

* **Input features:** 40 Mel-frequency features per time step
* **Recurrent layer:** GRU
* **GRU hidden size:** 32
* **Output layer:** Linear layer with one output
* **Loss function:** `BCEWithLogitsLoss`
* **Optimizer:** Adam
* **Learning rate:** `0.001`
* **Training epochs:** 10
* **Random seed:** 42
* **Device:** CUDA / NVIDIA GPU

## Audio Preprocessing

### 1. Word Segmentation

The original recordings contain multiple repetitions of the words.

Instead of manually splitting the recordings, an energy-based silence detection method is used.

The audio is divided into **20 ms frames**, and the mean absolute amplitude of each frame is calculated.

A frame is considered silent when its amplitude is below:

```text
silence threshold = 0.022
```

A silence must last for at least **100 ms** to be considered a separation between words.

Additional padding of **50 ms** is added around each detected word.

Very short segments below **100 ms** are discarded to remove noise or unwanted short segments.

### 2. Mel-spectrogram Features

Each extracted word is converted into a Mel-spectrogram using:

* FFT size: `1024`
* Hop length: `256`
* Number of Mel bins: `40`

The amplitude is converted to a logarithmic decibel scale using `AmplitudeToDB`.

The resulting feature representation has the form:

```text
(T, 40)
```

where:

* `T` = number of time frames
* `40` = number of Mel-frequency features

Because different recordings can have different durations, the time dimension can vary between samples.

## Dataset Split

The "Yes" and "No" samples are independently divided into:

* **80% training**
* **10% validation**
* **10% test**

The splits use `random_state=42` for reproducibility.

The labels are:

```text
Yes → 0
No  → 1
```

## Feature Normalization

Normalization is performed using statistics calculated **only from the training data**.

For every Mel-frequency feature:

```text
normalized = (x - mean) / std
```

The same training-set mean and standard deviation are then applied to the validation and test sets.

This avoids using information from the test set when calculating the normalization parameters.

## Training

The model is trained using **stochastic gradient descent at the sample level**.

For every epoch:

1. Training samples are shuffled
2. One audio sequence is passed through the GRU at a time
3. The prediction is compared with the ground-truth label
4. Backpropagation is performed
5. The Adam optimizer updates the model parameters

The training configuration is:

```text
Optimizer: Adam
Learning rate: 0.001
Loss: BCEWithLogitsLoss
Epochs: 10
```

## Results

The trained GRU classifier achieved:

| Metric        |   Result |
| ------------- | -------: |
| Test Accuracy | **100%** |

The test accuracy was calculated using `torchmetrics.Accuracy` with binary classification.

## Technologies Used

* Python
* PyTorch
* TorchAudio
* TorchMetrics
* scikit-learn
* CUDA
* GRU / RNN
* Mel-spectrograms

## Requirements

Install the required Python packages:

```bash
pip install torch torchaudio torchmetrics scikit-learn
```

A CUDA-compatible NVIDIA GPU is used for model training and inference in this project.

## Project Structure

```text
yes-no-classifier/
│
├── yes_no_classifier.ipynb
├── README.md
└── yes_new_38.wav
└── no_new_42.wav
```

The notebook contains the complete pipeline from audio preprocessing through model evaluation.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/vibhav-gupta-1989/yes_no_speech_classifier.git
cd yes_no_speech_classifier
```

2. Install the dependencies:

```bash
pip install torch torchaudio torchmetrics scikit-learn
```

3. Place the required audio recordings in the appropriate directory.

4. Open the notebook:

```bash
jupyter notebook yes_no_classifier.ipynb
```

5. Run the notebook cells sequentially.

## Key Learning Outcomes

This project provided hands-on experience with:

* Audio preprocessing using PyTorch/TorchAudio
* Silence detection and audio segmentation
* Mel-spectrogram feature extraction
* Variable-length sequential data
* Recurrent Neural Networks
* GRUs for sequence classification
* Binary classification with `BCEWithLogitsLoss`
* Training models on individual sequences
* GPU-accelerated training with CUDA
* Proper train/validation/test splitting
* Dataset-based feature normalization
* Model evaluation using TorchMetrics

## Future Improvements

Possible extensions to this project include:

* Increasing the size and diversity of the dataset
* Adding data augmentation such as background noise and time shifting
* Experimenting with different GRU hidden sizes
* Comparing GRU performance with LSTM and 1D CNN models
* Using mini-batch training with padding and masking
* Adding a confusion matrix and additional evaluation metrics
* Testing the model on recordings from different speakers
* Building a real-time "Yes"/"No" speech recognition application
