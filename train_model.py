import os
import json
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# ================= CONFIG =================
DATA_DIR = "data_split"

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16

EPOCHS_FROZEN = 15
EPOCHS_STAGE1 = 20   # more room — was hitting stride at epoch 15
EPOCHS_STAGE2 = 20

SAVE_PATH = "wheat_model_v2.keras"
SEED      = 42

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
print("Num classes:", num_classes)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

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
print("\nClass counts:", dict(zip(class_names, counts)))
print("Class weights:", class_weights)

steps_per_epoch = math.floor(total_samples / BATCH_SIZE)
print(f"Steps per epoch: {steps_per_epoch}")

# ───── Pipeline ─────
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (train_ds
    .cache()
    .shuffle(1000, seed=SEED)
    .repeat()
    .prefetch(AUTOTUNE)
)
val_ds  = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ───── Augmentation ─────
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.1),
], name="augmentation")

# ───── Build model ─────
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs  = keras.Input(shape=(*IMG_SIZE, 3))
x       = data_augmentation(inputs)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ───── Helpers ─────
def recompile(lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

def make_callbacks(monitor_metric="val_accuracy", mode="max"):
    """
    Monitor val_accuracy instead of val_loss.
    val_loss can plateau while val_accuracy is still climbing —
    which is exactly what happened in Phase 2b, causing premature stopping.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=6,          # more patience — give it room to improve
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode=mode,
            patience=4,          # was 3 — fired too early before
            factor=0.4,
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            SAVE_PATH,
            monitor=monitor_metric,
            mode=mode,
            save_best_only=True,
            verbose=1
        )
    ]

def run_phase(label, epochs, lr):
    print(f"\n=== {label} (lr={lr}) ===")
    recompile(lr)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weights,
        callbacks=make_callbacks()
    )
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"    → Val accuracy after {label}: {val_acc:.4f}")
    return val_acc

# ─────────────────────────────────────────
# PHASE 1 — Head only, base fully frozen
# ─────────────────────────────────────────
base_model.trainable = False
acc1 = run_phase("Phase 1: Head only", EPOCHS_FROZEN, lr=1e-3)

# ─────────────────────────────────────────
# PHASE 2a — Unfreeze last 20 layers
# ─────────────────────────────────────────
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
acc2a = run_phase("Phase 2a: Last 20 layers", EPOCHS_STAGE1, lr=5e-5)

# ─────────────────────────────────────────
# PHASE 2b — Unfreeze last 40 layers
# ─────────────────────────────────────────
for layer in base_model.layers[:-40]:
    layer.trainable = False
acc2b = run_phase("Phase 2b: Last 40 layers", EPOCHS_STAGE2, lr=1e-5)

# ───── Summary ─────
print("\n=== Phase Accuracy Summary ===")
print(f"  Phase 1  (head only):    {acc1:.4f}")
print(f"  Phase 2a (last 20):      {acc2a:.4f}")
print(f"  Phase 2b (last 40):      {acc2b:.4f}")
print(f"  Best val accuracy:       {max(acc1, acc2a, acc2b):.4f}")

# ───── Final evaluation ─────
print("\n=== Final Test Accuracy ===")
model.evaluate(test_ds)

print("\nModel saved as:", SAVE_PATH)