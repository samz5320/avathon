# avathon
# CAPTCHA Recognition Model

This project implements a Convolutional Neural Network (CNN) to recognize CAPTCHA images. The model is trained to predict a sequence of digits from CAPTCHA images.

## Model Architecture

The model is implemented using PyTorch and consists of the following layers:

- **Convolutional Layers:**
  - Conv2d with 32 filters, kernel size 3, followed by BatchNorm2d and ReLU activation, and MaxPool2d.
  - Conv2d with 64 filters, kernel size 3, followed by BatchNorm2d and ReLU activation, and MaxPool2d.
  - Conv2d with 128 filters, kernel size 3, followed by BatchNorm2d and ReLU activation, and MaxPool2d.

- **Fully Connected Layers:**
  - Flatten layer followed by Linear layer with 512 units, BatchNorm1d, ReLU activation, and Dropout.
  - Linear layer to output the final predictions.

The model is designed to predict a sequence of 6 digits, each digit being one of 10 possible classes (0-9).

## Training and Validation

The model is trained using the following configuration:

- **Batch Size:** 64
- **Epochs:** 100
- **Learning Rate:** 1e-4
- **Device:** CUDA if available, otherwise CPU

### Metrics

The model's performance is evaluated using two metrics:

- **Per-digit Accuracy:** Measures the accuracy of predicting each digit in the sequence.
- **Full-sequence Accuracy:** Measures the accuracy of predicting the entire sequence correctly.

During training, the model achieved the following metrics:

Per-digit acc: 0.9396 | Full-seq acc: 0.7480

## Inference

To perform inference, the model processes an input image through the following steps:

1. **Preprocessing:** Resize the image to (50, 200), convert to tensor, and normalize.
2. **Prediction:** The model outputs a sequence of predicted digits.

