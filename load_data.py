import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set paths to your data folders
train_path = "dataset/train"
val_path = "dataset/val"
test_path = "dataset/test"

# Common parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Print class names to confirm labels
class_names = train_ds.class_names
print("Class Names:", class_names)

# Apply MobileNetV2 preprocessing
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# Optional: Use prefetching to speed up training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

