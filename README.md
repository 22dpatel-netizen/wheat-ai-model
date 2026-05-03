Wheat Disease AI

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
wheat_ai/

train_model_v2.py
continue_training.py

test_image.py
(This runs the model to predict the disease for 'test.jpg')

wheat_model_v2.keras
(The trained AI model)

data/

train/
val/
test/
Installation

Install dependencies:

pip install tensorflow numpy pillow
How to Run
Open terminal in the project folder
Run training:
python train_model_v2.py
After training, place an image named test.jpg in the root folder
Run testing:
python test_image.py
Results

The model typically achieves around 78–85% validation accuracy, depending on dataset quality and balance.

Notes
The model automatically saves the best version during training
Training can be continued using continue_training.py
Dataset must be organised into train, val, and test folders with matching class subfolders
Future Improvements
Improve dataset balance across classes
Refine similar disease groups to reduce confusion
Deploy as a simple application for image upload and prediction (user friendly for farmers)
Author

Team Moloch
Raspberry Pi PaPi Challenge Team
