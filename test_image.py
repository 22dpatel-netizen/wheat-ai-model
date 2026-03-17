import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# -----------------------
# SETTINGS
# -----------------------

MODEL_PATH = "best_wheat_model.keras"
IMAGE_PATH = "test.jpg"
IMG_SIZE = (224, 224)

# These MUST match your training classes
CLASS_NAMES = ["blight", "healthy", "na", "pest", "rust"]

# -----------------------
# LOAD MODEL
# -----------------------

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------
# LOAD IMAGE
# -----------------------

img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize(IMG_SIZE)

img_array = np.array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# -----------------------
# PREDICT
# -----------------------

prediction = model.predict(img_array)[0]

predicted_class = CLASS_NAMES[np.argmax(prediction)]
confidence = np.max(prediction)

print("\nPrediction:", predicted_class)
print("Confidence: {:.2f}%".format(confidence * 100))
