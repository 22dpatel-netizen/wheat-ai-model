import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
import os

# ======================
# SETTINGS
# ======================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5

TRAIN_DIR = "data/train"
VAL_DIR = "data/valid"

MODEL_PATH = "best_wheat_model.keras"

# ======================
# LOAD DATA
# ======================

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# ======================
# LOAD OR CREATE MODEL
# ======================

if os.path.exists(MODEL_PATH):
    print("\n🔁 Loading existing trained model...\n")
    model = tf.keras.models.load_model(MODEL_PATH)

else:
    print("\n🆕 Creating new model...\n")

    base_model = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

# ======================
# COMPILE MODEL
# ======================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# SAVE BEST MODEL
# ======================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ======================
# TRAIN MODEL
# ======================

print("\n🚀 Training started...\n")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("\n✅ Training finished.")
print("Best model saved as:", MODEL_PATH)
