import os
import json
import numpy as np
from tensorflow import keras

# ================= CONFIG =================
MODEL_PATH = "wheat_model_v2.keras"
IMG_SIZE   = (224, 224)
# ==========================================

# ───── Find test image ─────
if os.path.exists("test.jpg"):
    img_path = "test.jpg"
elif os.path.exists("test.png"):
    img_path = "test.png"
else:
    print(" No test image found. Place a 'test.jpg' or 'test.png' in this folder.")
    exit()

print(f"📸 Using image: {img_path}")

# ───── Load class names ─────
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ───── Load model ─────
print(f"🤖 Loading model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

# ───── Load and preprocess image ─────
img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# ───── Predict ─────
predictions = model.predict(img_array, verbose=0)
probabilities = predictions[0] * 100

predicted_index = np.argmax(probabilities)
predicted_class = class_names[predicted_index]
confidence      = probabilities[predicted_index]

# ───── Results ─────
print("\n" + "="*40)
print(f"  🌾 Prediction : {predicted_class}")
print(f"  📊 Confidence : {confidence:.1f}%")
print("="*40)

print("\n  All class probabilities:")
for cls, prob in sorted(zip(class_names, probabilities), key=lambda x: -x[1]):
    bar = "█" * int(prob / 5)
    print(f"  {cls:<12} {prob:>6.1f}%  {bar}")
