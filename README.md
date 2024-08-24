# DekaBank-BERT-Transformer: Fine-Tuning and Evaluation
(From a Hackathon hosted by Deka)
## Overview

This repository contains a Jupyter Notebook that demonstrates the fine-tuning and evaluation of a BERT-based model for sequence classification tasks using PyTorch and Hugging Face's Transformers library. The notebook walks through the process of training a BERT model on a custom dataset, monitoring the training process, and evaluating model performance using accuracy metrics.

This notebook is from my participation at a workshop event at Deka, and was intended for educational purposes only.

## Contents

1. **Setup and Configuration**
2. **Data Preparation**
3. **Model Training**
4. **Model Evaluation**
5. **Results and Analysis**

## Setup

### Prerequisites

Ensure you have the following Python packages installed:

- `numpy`
- `pandas`
- `torch`
- `transformers`
- `keras`
- `matplotlib`
- `scikit-learn`

### Configuration

1. **Reproducibility**:
   The notebook sets seeds for `random`, `numpy`, and `torch` to ensure reproducibility across different runs. The seed value used is `42`.

2. **Data Preparation**:
   The notebook assumes you have a dataset ready for training and validation. The data should be preprocessed and loaded into PyTorch DataLoaders.

## Usage

### Running the Notebook

1. **Load the Notebook**:
   Open the Jupyter Notebook file (`.ipynb`) in your Jupyter environment.

2. **Adjust Hyperparameters**:
   Modify parameters such as the number of epochs, learning rate, and batch size according to your needs.

3. **Execute Cells**:
   Run all the cells in the notebook to execute the training and evaluation pipeline.

### Code Sections

1. **Data Preparation**:
   - **Tokenization**: Convert text data into input IDs and attention masks suitable for BERT.
   - **DataLoaders**: Create PyTorch DataLoaders for batching and shuffling training and validation data.

2. **Model Training**:
   - **Initialization**: Load a pre-trained BERT model and configure the optimizer and learning rate scheduler.
   - **Training Loop**: Perform fine-tuning by iterating over the training data, calculating loss, and updating model parameters using backpropagation.

3. **Model Evaluation**:
   - **Validation**: Evaluate the model on the validation dataset to measure performance.
   - **Accuracy Calculation**: Use a custom accuracy function to compute the proportion of correct predictions.

## Results and Analysis

- **Training Loss**: Monitored during each epoch to observe the convergence of the model.
- **Validation Accuracy**: Calculated at the end of each epoch to evaluate the model's performance on unseen data.
- **Matthews Correlation Coefficient (MCC)**: In the end we managed to achieve a MCC of **~0.375** which is decent considering we didnt do any hyperparamater tuning.

## Acknowledgments

- **Hugging Face**: For providing the BERT model and Transformers library.
- **PyTorch**: For its deep learning framework that enables efficient training and evaluation.
- Deka
- [Chris McCormick's Blog](https://github.com/chrisjmccormick)
