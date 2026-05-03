# Wheat Disease AI 🌾

This project uses a deep learning model to classify wheat health and detect common diseases from images. It is designed as a practical application of computer vision that could assist with early crop monitoring.

---

## Overview

The model is built using TensorFlow and transfer learning with EfficientNetB0. It analyses images of wheat and predicts one of several categories including disease, pests, or healthy crops. A multi-phase fine-tuning strategy is used to maximise accuracy while preventing the model from forgetting its pre-trained knowledge.

---

## Features

- Image classification using a convolutional neural network
- Transfer learning with EfficientNetB0 (pre-trained on ImageNet)
- Trained on ~13,000 labelled images across 6 classes
- Multi-phase gradual fine-tuning for stable, high accuracy training
- Class weights to handle imbalanced datasets
- Data augmentation to improve generalisation
- Automatic saving of the best performing model based on validation accuracy
- Separate continue training script to refine an existing model

---

## Classes

- Blight
- Healthy
- Mildew
- Pests
- Rust
- Smut

---

## Project Structure

```
wheat_ai/
├── train_model.py           # Main training script (3-phase fine-tuning)
├── continue_training.py     # Continue training from an existing saved model
├── diagnose_dataset.py      # Check class counts and dataset split health
├── resplit_dataset.py       # Re-split data into correct train/val/test ratios
├── test_image.py            # Run the model on a single image (test.jpg)
├── wheat_model_v2.keras     # Best saved model
├── class_names.json         # Saved class labels
├── test.jpg                 # Place your test image here
└── data_split/
    ├── train/               # Training images (~75% of data)
    ├── val/                 # Validation images (~15% of data)
    └── test/                # Test images (~10% of data)
```

---

## Installation

Install dependencies:

```
pip install tensorflow numpy pillow
```

---

## How to Run

Open a terminal in the project folder.

**Check your dataset is healthy before training:**
```
python diagnose_dataset.py
```

**Re-split your data into correct ratios (run once on your raw data):**
```
python resplit_dataset.py
```

**Run training:**
```
python train_model.py
```

**Continue training an existing model:**
```
python continue_training.py
```

**Test on a single image — place an image named `test.jpg` in the root folder, then run:**
```
python test_image.py
```

---

## Results

The model achieves **~89% test accuracy** across 6 classes using a 3-phase gradual fine-tuning approach:

| Phase | Layers Unfrozen | Val Accuracy |
|---|---|---|
| Phase 1 — Head only | 0 (base frozen) | 86.6% |
| Phase 2a — Fine-tune top 20 | Last 20 layers | 88.8% |
| Phase 2b — Fine-tune top 40 | Last 40 layers | 89.3% |

---

## Notes

- The model automatically saves the best version during training based on validation accuracy
- Training uses class weights to fairly handle imbalanced class sizes
- Data augmentation (flipping, rotation, zoom, brightness) is applied during training only
- The dataset should be organised into `data_split/train`, `data_split/val`, and `data_split/test` folders with class subfolders
- Do not unfreeze the full model during fine-tuning — this causes catastrophic forgetting and drops accuracy significantly

---

## Future Improvements

- Expand dataset size per class to push accuracy above 92%
- Add more disease categories and crop types
- Deploy as a mobile or web application for farmers to upload images in the field
- Implement Test-Time Augmentation (TTA) for higher confidence predictions

---

## Author

**Team Moloch** — Raspberry Pi PaPi Challenge
