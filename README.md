# Wheat Disease AI 🌾

This project uses a deep learning model to classify wheat health and detect common diseases from images. It is designed as a practical
application of computer vision that could assist with early crop monitoring.

## Overview

The model is built using TensorFlow and transfer learning with EfficientNetV2B0. It analyses images of wheat and predicts one of several
categories including disease, pests, healthy crops, or non-wheat inputs.

## Features

* Image classification using a convolutional neural network
* Transfer learning with EfficientNetV2B0
* Trained on 14,000+ custom images
* Supports multiple disease and pest categories
* Includes a non-wheat class for real-world reliability
* Automatic saving of the best performing model

## Classes

* blight
* healthy
* pest
* rust
* na

## Project Structure

wheat_ai/ (folder)

* train_wheat_ai.py 
* test_image.py (running this will use the ai model to predict the disease for the image called 'test.jpg' this must be inside the folder aswell)
* best_wheat_model.keras (This is the best ai model used to analyse photso)
* data/

  * train/
  * valid/

## Installation

* install dependencies: pip install tensorflow numpy pillow

## How to Run

* open terminal in the project folder
* run training: python train_wheat_ai.py
* after training completes, place an image named test.jpg in the root folder
* run testing: python test_image.py

## Results

The model typically achieves around 75–85 percent validation accuracy depending on dataset balance and size.

## Notes

* the model automatically saves the best version during training
* training can be continued by rerunning the script
* dataset must be organised into train and valid folders with class subfolders

## Future Improvements

* improve dataset balance across classes
* expand to more disease categories and crop types
* deploy as a simple application for image upload and prediction (user friendly for farmers)

## Author

Team Moloch - Raspberry Pi PaPi challenge team name.
