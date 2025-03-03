# Paddy Disease Classification

This project classifies paddy leaf diseases using EfficientNet, Vision Transformer (ViT), and CNN models trained on the Paddy Doctor dataset and Rice Leaf Disease Classification dataset. The frontend allows users to upload images and receive predictions using Gradio.

## Features
- Classifies 9 disease classes and 1 normal class.
- Utilizes deep learning models with transfer learning.
- Implements data augmentation and optimized hyperparameters.
- Provides a user-friendly frontend using Gradio for real-time predictions.

## Installation

1. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

## Model Training (Optional)
To retrain the models: Run the leaf-disease.ipynb file

Ensure the dataset is available in the appropriate directory.
