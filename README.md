Wheat Disease AI 🌾

This project uses a deep learning model to classify wheat health and detect common diseases from images. It is designed as a practical application of computer vision that could assist with early crop monitoring.

Overview

The model is built using TensorFlow and transfer learning with EfficientNetB0. It analyses images of wheat and predicts one of several categories including disease, pests, or healthy crops.

Features

Image classification using a convolutional neural network
Transfer learning with EfficientNetB0
Trained on 14,000+ custom images
Supports multiple disease and pest categories
Automatic saving of the best performing model

Classes

Healthy
Rust
Blight
Mildew
Pests
Root_Rot
Smut

Project Structure
wheat_ai/ (folder)

train_model_v2.py

continue_training.py

test_image.py (running this will use the ai model to predict the disease for the image called 'test.jpg' this must be inside the folder as well)

wheat_model_v2.keras (This is the best AI model used to analyse photos)

data/

train/
val/
test/
Installation

install dependencies: pip install tensorflow numpy pillow

How to Run

open terminal in the project folder
run training: python train_model_v2.py

after training completes, place an image named test.jpg in the root folder
run testing: python test_image.py

Results

The model typically achieves around 78–85 percent validation accuracy depending on dataset quality and balance.

Notes

the model automatically saves the best version during training
training can be continued using continue_training.py
dataset must be organised into train, val, and test folders with matching class subfolders

Future Improvements

improve dataset balance across classes
refine similar disease groups to reduce confusion
deploy as a simple application for image upload and prediction (user friendly for farmers)

Author

Team Moloch - Raspberry Pi PaPi challenge team name.
