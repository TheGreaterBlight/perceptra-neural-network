# Perceptra Neural Network - Image Classification Engine
## Overview
This project is a 4-3-1 Multilayer Perceptron (MLP) designed for image classification, developed for the Artificial Intelligence II Seminar (2025B). The system was built from the ground up in pure JavaScript, without the use of external machine learning libraries like TensorFlow or PyTorch. This "from-scratch" approach was chosen to ensure total control over the learning process and a deep understanding of neural network mechanics.

Team Members
Carlos Eduardo Avila Bautista - Computer Engineering
Felix Eduardo Estrada Huerta - Computer Engineering

Objectives
Develop a 4-3-1 multilayer neural network architecture for image classification.

Implement manual backpropagation and weight/bias optimization logic.

Demonstrate the network's capability to differentiate between Iris setosa and Iris versicolor based on visual features.

## Technical Architecture
The network is structured to process numerical features extracted from image data through a custom computer vision engine.

Input Layer: 4 neurons representing flower characteristics (sepal and petal length and width).

Hidden Layer: 3 neurons utilizing a Sigmoid activation function.

Output Layer: 1 neuron providing binary classification between the two target species.

## Mathematical Framework
The implementation utilizes specific matrices and vectors for the learning process:

Hidden Weight Matrix: 4 × 3 dimensions.

Hidden Bias Vector: Size 3.

Output Weight Matrix: 1 × 3 dimensions.

Output Bias: Scalar value.

## Core Implementation Details
The project consists of several critical modules responsible for the intelligence engine:

Activation Functions: Manual implementation of sigmoid() and sigmoidDerivative() to facilitate non-linear processing and error correction.

Backpropagation: The trainStep() function handles the backward pass, calculating gradients and updating weights to minimize the error.

Feature Extraction: The extractFeaturesFromImage() module acts as the vision engine, processing raw pixels to generate normalized inputs for the network.

Interactive Learning: The trainWithFeedback() function allows for supervised learning where the system can adapt based on real-time user corrections.

## Performance and Results
After 500 epochs of training, the network achieved approximately 95% accuracy on the training dataset. The model successfully classifies real images of flowers and demonstrates a high level of confidence that increases throughout the training process. The project successfully validates the effectiveness of manually implemented gradient descent and matrix multiplication in a browser-based environment.

## Bibliography
Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.). O’Reilly.

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386–408.

Abadi, M., et al. (2016). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. Google.

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179–188.
