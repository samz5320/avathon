# avathon
# CAPTCHA Recognition Model

This project implements a CRNN (Convolutional Recurrent Neural Network) model using PyTorch to recognize 6-digit numeric CAPTCHA images. The model is trained to predict a sequence of digits from distorted grayscale CAPTCHA images using CTC (Connectionist Temporal Classification) loss.

## Model Architecture

The model is implemented using PyTorch and consists of the following layers:

- **Convolutional Layers:**
  - Conv2d with 64 filters, kernel size 3, ReLU activation, MaxPool2d
  - Conv2d with 128 filters, kernel size 3, ReLU activation, MaxPool2d

- **Recurrent Layer:**
  - 2-layer bidirectional LSTM with hidden size 128

- **Fully Connected Layer:**
  - Linear layer with output dimension equal to the number of classes (10 digits + CTC blank = 11)


## Training and Validation

The model is trained using the following configuration:

- **Batch Size:** 32  
- **Epochs:** 50
- **Learning Rate:** 1e-3  
- **Loss Function:** CTC Loss  
- **Optimizer:** Adam  
- **Device:** CUDA if available, otherwise CPU

Training includes logging, model checkpointing, and evaluation after each epoch.

### Metrics

The model's performance is evaluated using two metrics:

- **Per-digit Accuracy:** Measures accuracy of individual digit predictions.
- **Full-sequence Accuracy:** Measures accuracy of correctly predicting the entire 6-digit string.

Per-digit acc: 0.9933 | Seq acc: 0.9690  

[PREDICT] image_validation_1.png → 001279