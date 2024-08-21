# XRayAnalyzerPython

## Description
XRayAnalyzerPython is an application designed for the automatic analysis of pulmonary X-ray images, classifying them binary into two categories: Normal or Pneumonia. This innovative solution uses artificial intelligence techniques to assist in the rapid and accurate diagnosis of pneumonia.

## Key Features
- Uses the DenseNet121 model for image classification
- Provides an intuitive graphical interface for uploading and analyzing X-rays
- Displays classification results along with associated probabilities
- Processes images up to 10 times faster than traditional methods
- Achieves a diagnostic accuracy of up to 95%

## Technologies Used
- Python
- PyTorch
- DenseNet121 (Convolutional Neural Network)
- GUI: PyQt5

## Ongoing Research
We are currently working on an article comparing the performance of multiple convolutional neural network (CNN) architectures and artificial intelligence solutions in the context of pulmonary X-ray analysis. This study will provide a comprehensive perspective on the effectiveness of different approaches in the field of AI-assisted diagnosis.

## Application Interface

### Example: Healthy Patient
![Healthy Patient](https://github.com/user-attachments/assets/84299d26-87e2-4510-bec5-51ae5e055d31)
This image shows the application interface for a patient diagnosed as healthy. The analysis results can be seen in the right panel.

### Example: Patient with Pneumonia
![Patient with Pneumonia](https://github.com/user-attachments/assets/32bb06d2-e4bb-42fb-a351-6ae8ea78e293)
This image illustrates the application interface for a patient diagnosed with pneumonia. The analysis results are displayed in the right panel.

## Installation and Usage
Run the file widgey.py with the following requirements:
torch>=1.7.0
torchvision>=0.8.1
numpy>=1.18.5
Pillow>=8.0.1
matplotlib>=3.3.2
scikit-learn>=0.23.2
pandas>=1.1.4
opencv-python>=4.4.0.46
PyQt5>=5.15.2

## Availability of Training Code
The training code for the DenseNet121 model used in this project will be made public soon. This will allow users and developers to explore, modify, and contribute to the ongoing improvements of XRayAnalyzerPython. Stay tuned for updates.

## Contact
For more information or questions, please contact [Mihai Bundea](https://ro.linkedin.com/in/mihai-bundea-551a55225)
