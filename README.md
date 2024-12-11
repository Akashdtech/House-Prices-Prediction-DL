# House-Prices-Prediction-DL
This project implements an audio classification pipeline for classifying animal sounds (bird, cat, dog) from a dataset. The pipeline processes audio files, extracts features, and applies machine learning and deep learning models to classify the sounds. The workflow includes:

    Data Loading and Extraction:
        The dataset is extracted from a zip file containing audio files for bird, cat, and dog sounds.

    Data Visualization:
        Sample audio files are loaded using librosa and visualized as waveforms and spectrograms for each class (bird, cat, dog).

    Feature Extraction:
        MFCC (Mel-frequency cepstral coefficients) features are extracted from the audio files using librosa to represent audio characteristics.
        The extracted features are scaled and stored in a dataframe with corresponding labels (bird = 0, dog = 1, cat = 2).

    Data Preprocessing:
        The dataset is split into training and testing sets (80% training, 20% testing).
        Standard scaling is applied to the features.

    Model Training and Evaluation:
        MLP Model:
            A multi-layer perceptron (MLP) model is defined and trained on the scaled features.
            Performance is evaluated using accuracy, classification report, and confusion matrix.
        CNN Model:
            A Convolutional Neural Network (CNN) model is defined by reshaping the features and training them with convolutional layers.
            Performance is similarly evaluated using accuracy, classification report, and confusion matrix.

    Performance Metrics:
        Accuracy, classification report, and confusion matrix are generated for both models to assess and compare performance.

This repository provides a comprehensive approach to audio classification using both MLP and CNN models, demonstrating feature extraction, model training, and evaluation for classifying animal sounds.
