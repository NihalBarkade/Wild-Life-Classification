import os
import shutil
import random

# Path to original dataset folder
original_dataset_dir = "dataset"  # adjust path if needed

# Desired split percentages
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Get all class names (folder names)
classes = [d for d in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, d))]

# Create directories
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(original_dataset_dir, split, cls), exist_ok=True)

# Move files to respective folders
for cls in classes:
    class_dir = os.path.join(original_dataset_dir, cls)
    if not os.path.isdir(class_dir):
        continue
    images = os.listdir(class_dir)
    random.shuffle(images)
    
    train_end = int(len(images) * train_split)
    val_end = train_end + int(len(images) * val_split)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    for file_set, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        for img_file in file_set:
            src = os.path.join(class_dir, img_file)
            dst = os.path.join(original_dataset_dir, split, cls, img_file)
            shutil.move(src, dst)

    # Optional: Remove empty original class folder
    os.rmdir(class_dir)

print("Dataset split complete.")
