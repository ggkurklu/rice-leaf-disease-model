# Rice Leaf Disease Classification using CNN

A Convolutional Neural Network (CNN) model for classifying diseases in rice leaves. This model helps in early detection of common rice diseases such as Bacterial Blight, Brown Spot, and Leaf Smut.

## Dataset
The model is trained on a dataset containing labeled images of healthy and diseased rice leaves. Example datasets:
- [Rice Leaf Disease Dataset from Kaggle](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)
- Custom dataset with augmented images.

## Model Architecture
- **Input Layer**: 224x224x3 (RGB images)
- **Convolutional Layers**: Multiple Conv2D + MaxPooling2D layers with ReLU activation.
- **Dense Layers**: Fully connected layers with Dropout for regularization.
- **Output Layer**: Softmax activation for multi-class classification.

## Requirements
- Python 3.8+
- TensorFlow 2.x / Keras
- OpenCV (for image preprocessing)
- NumPy, Pandas, Matplotlib

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rice-leaf-disease-cnn.git
   cd rice-leaf-disease-cnn