import os
import math
import tensorflow as tf
from tensorflow import keras

# ================= CONFIG =================
DATA_DIR   = "data_split"
MODEL_PATH = "wheat_model_v2.keras"   # model to load
SAVE_PATH  = "wheat_model_v3.keras"   # always save to a NEW version

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS_EXTRA = 20

SEED = 42

tf.config.optimizer.set_jit(True)
# ==========================================

# ───── Load datasets ─────
train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="categorical"
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="categorical"
)

test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)

# ───── Class weights ─────
counts = [
    len(os.listdir(os.path.join(DATA_DIR, "train", cls)))
    for cls in class_names
]
total_samples = sum(counts)
class_weights = {
    i: total_samples / (num_classes * count)
    for i, count in enumerate(counts)
}
print("Class weights:", class_weights)

steps_per_epoch = math.floor(total_samples / BATCH_SIZE)
print(f"Steps per epoch: {steps_per_epoch}")

# ───── Pipeline optimisation ─────
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ───── Load existing model ─────
model = keras.models.load_model(MODEL_PATH)
print("\nLoaded model:", MODEL_PATH)

# Very small LR — we're polishing, not relearning
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ───── Callbacks ─────
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        patience=2,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        SAVE_PATH,
        save_best_only=True,
        verbose=1
    )
]

# ───── Continue training ─────
print("\n=== Continuing Training ===")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_EXTRA,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights,
    callbacks=callbacks
)

# ───── Evaluate ─────
print("\n=== Final Test Accuracy ===")
model.evaluate(test_ds)

print("\nNew model saved as:", SAVE_PATH)