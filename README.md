# Dog vs Cat Classification using ResNet Transfer Learning

This project implements a complete dog vs cat image classification system using ResNet50 transfer learning with TensorFlow/Keras. The implementation includes comprehensive training, evaluation, and visualization capabilities in a clean Python script format.

## Features

- **Transfer Learning**: Uses pre-trained ResNet50 for feature extraction
- **Data Augmentation**: Includes rotation and horizontal flipping
- **Comprehensive Evaluation**: Confusion matrix, classification report, and sample predictions
- **Training Visualization**: Plots training history, accuracy, and loss curves
- **Fine-tuning**: Optional fine-tuning of top ResNet layers for better performance
- **Model Persistence**: Save and load trained models
- **Progress Tracking**: Real-time training progress and callbacks

## Files Structure

```
├── dog_cat_classifier.py      # Main training script
├── setup_and_run.py          # Setup and environment preparation
├── requirements.txt          # Python dependencies
├── Dog_cat_classification_v2.ipynb  # Original Jupyter notebook
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment

First, run the setup script to install dependencies and check your environment:

```bash
python setup_and_run.py
```

### 2. Data Preparation

Download the Kaggle Dogs vs Cats dataset:
1. Go to [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Download `train.zip`
3. Extract it to create a `train/` folder in your project directory
4. The folder should contain files like `dog.1.jpg`, `cat.1.jpg`, etc.

Expected structure:
```
train/
├── dog.1.jpg
├── dog.2.jpg
├── cat.1.jpg
├── cat.2.jpg
└── ... (25,000 total images)
```

### 3. Run Training

Execute the main training script:

```bash
python dog_cat_classifier.py
```

## Training Process

The script performs the following steps:

1. **Data Loading**: Loads and preprocesses images from the train directory
2. **Data Splitting**: Splits data into train (60%), validation (20%), and test (20%) sets
3. **Model Creation**: Creates ResNet50-based transfer learning model
4. **Initial Training**: Trains with frozen ResNet weights
5. **Evaluation**: Evaluates on test set and generates visualizations
6. **Fine-tuning**: Unfreezes top ResNet layers for additional training
7. **Final Evaluation**: Final performance assessment

## Model Architecture

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Input Size**: 224x224x3 RGB images
- **Data Augmentation**: Random horizontal flip and rotation
- **Custom Layers**: 
  - Global Average Pooling
  - Dense layer (128 units, ReLU)
  - Dropout (0.2)
  - Output layer (2 units, Softmax)

## Output Files

The script generates several output files:

- `training_history.png`: Training and validation accuracy/loss curves
- `confusion_matrix.png`: Confusion matrix heatmap
- `sample_predictions.png`: Sample predictions with confidence scores
- `dog_cat_classifier_final.h5`: Trained model file

## Configuration

You can modify the following parameters in the `main()` function:

- `data_path`: Path to your training images (default: 'train/')
- `image_size`: Input image dimensions (default: (224, 224))
- `batch_size`: Training batch size (default: 32)
- `epochs`: Number of training epochs (default: 20)

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

Install all requirements with:
```bash
pip install -r requirements.txt
```

## License

This project is open source and available under the MIT License.
